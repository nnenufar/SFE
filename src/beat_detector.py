from scipy import signal
import librosa
from scipy.signal import windows
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from praatio import textgrid
import warnings
from sklearn.preprocessing import MaxAbsScaler
from feat_extractor import MFCC

class BD():
  def __init__(self, sr, bandpass_left, bandpass_right, bandpass_order = 1, lowpass_order = 3, lowpass_cutoff = 10, min_dur = 0.05, min_peak_prominence = 0.25):
    self.bandpass_order = bandpass_order
    self.bandpass_left = bandpass_left
    self.bandpass_right = bandpass_right
    self.lowpass_order = lowpass_order
    self.lowpass_cutoff = lowpass_cutoff
    self.min_dur = min_dur
    self.min_peak_prominence = min_peak_prominence
    self.sr = sr
    self.nyquist = sr / 2
    self.feat_extractor = MFCC(sr = self.sr, n_mfcc = 13, frame_length = 512) #TODO: (?) adjust frame length

  def bandpass(self):
    '''
    First filter. Used to eliminate fricative noise and F0 energy,
    leaving energy in the formants region intact.
    '''
    bandpass_right = self.bandpass_right / self.nyquist
    bandpass_left = self.bandpass_left / self.nyquist
    b, a = signal.butter(self.bandpass_order, [bandpass_left, bandpass_right], btype = 'bandpass')

    # Filter delay
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", UserWarning)
      warnings.simplefilter("ignore", RuntimeWarning)
      w, gd_samples = signal.group_delay((b, a), fs=self.sr)
    
    passband_indices = np.where((w >= self.bandpass_left) & (w <= self.bandpass_right))
    delay_in_passband = gd_samples[passband_indices]
    mean_delay_samples = np.mean(delay_in_passband)
    self.delay_bp_samples = mean_delay_samples

    return (b, a)

  def rectify(self, waveform):
    '''
    Signal rectification using absolute values.
    '''
    return np.abs(waveform)

  def lowpass(self):
    '''
    Second filter. Used to produce a smooth amplitude envelope.
    '''
    cutoff = self.lowpass_cutoff / self.nyquist
    b, a = signal.butter(self.lowpass_order, cutoff, btype = 'lowpass')

    # Filter delay
    w, gd_samples = signal.group_delay((b, a), fs=self.sr)
    passband_indices = np.where(w <= self.lowpass_cutoff)
    delay_in_passband = gd_samples[passband_indices]
    mean_delay_samples = np.mean(delay_in_passband)
    self.delay_lp_samples = mean_delay_samples

    return (b, a)

  def detect_beats(self, waveform):
    '''
    Core function. Performs all signal processing steps.
    outputs a tuple:
    (beat_frames, envelope_spectrum) 
    '''
    # Waveform and intermediate signals are stored as instance attributes
    self.waveform = waveform
    self.tmax_s = len(self.waveform) / self.sr
    scaler = MaxAbsScaler()

    self.filtered = signal.lfilter(*self.bandpass(), self.waveform)
    self.rectified = self.rectify(self.filtered)
    self.envelope = signal.lfilter(*self.lowpass(), self.rectified)
    self.filter_delay = self.delay_bp_samples + self.delay_lp_samples

    # Compensate for filter delay
    self.envelope = self.envelope[int(self.filter_delay):]
    # Moving average smoothing (50ms window)
    self.envelope = pd.Series(self.envelope).rolling(int(self.sr/20), center=False, min_periods=1).mean().to_numpy()
    # Windowing (prevent spectral leakage)
    self.N = len(self.envelope)
    tukey = windows.tukey(self.N, alpha=0.05)
    self.envelope = self.envelope * tukey
    # Subtract mean from entire envelope to remove DC component in the spectrum, then normalize to unit variance
    self.envelope = (self.envelope - np.mean(self.envelope)) / np.std(self.envelope)

    self.derivative = np.gradient(self.envelope)
    self.derivative = scaler.fit_transform(self.derivative.reshape(-1, 1)).reshape(-1)
    self.peaks, _ = signal.find_peaks(self.derivative, distance = self.sr * self.min_dur, prominence = self.min_peak_prominence)

    #Detect beats
    # Note: self.beats will store the frame indices of the detected beats
    self.beats = []
    self.feats = []
    self.beat_intervals = []

    for index in self.peaks:
      peak_index = index
      if self.derivative[peak_index] > 0:
        self.beats.append(peak_index)

        #TODO: extract waveform spectrum at beat locations ()

        feats = self.feat_extractor.extract_mfcc(self.waveform, peak_index)
        self.feats.append(feats[:, 0])

      else:
        continue
      
    for idx, _ in enumerate(self.beats):
      if idx != 0:
        interval = (self.beats[idx] - self.beats[idx - 1]) / self.sr
        self.beat_intervals.append(interval)
      else:
        self.beat_intervals.append(0)
        

    beat_frames =  self.beats
    beat_intervals = self.beat_intervals
    feats = self.feats # len(beat_frames), n_mfcc

    # Extract envelope features
      # Spectrum of the magnitude envelope
    assert not np.any(np.isnan(self.envelope)), "Envelope contains NaN values"
    assert not np.any(np.isinf(self.envelope)), "Envelope contains infinite values"

    envSpec = np.fft.fft(self.envelope)
    self.N= len(envSpec)
    freq_axis = np.fft.fftfreq(self.N, 1 / self.sr)
    
    envSpec = abs(envSpec[:self.N// 2])
    freq_axis = freq_axis[:self.N// 2]

    self.spectrum = envSpec
    self.envSpecFreq = freq_axis

    return {'beat_frames':beat_frames, 'envelope_spectrum':envSpec, 'features':feats, 'beat_intervals':beat_intervals}
  
  def plot_ground_truth(self, ground_truth, plot):
      tg = textgrid.openTextgrid(ground_truth, includeEmptyIntervals=True)
      gt_tier = tg.getTier('beats')
      gt_timestamps = [entry.time * self.sr for entry in gt_tier.entries]
      for i, beat in enumerate(gt_timestamps):
        label = "Ground truth" if i == 0 else ""
        plot.axvline(beat, color='red', linestyle='-', alpha=0.7, label = label)

  def plot_filtered(self, out_file, ground_truth = None):
    '''
    Plotting resources. Must be called after detect_beats().
    Shows the original and bandpass filtered waveforms, the amplitude envelope, its derivative
    with peaks and beats marks, and the frequency response of the bandpass filter.
    '''

    fig = plt.figure(figsize=(12, 8))

    # Create 6 subplots, with the first 5 sharing x-axis
    ax1 = plt.subplot(6, 1, 1)
    ax2 = plt.subplot(6, 1, 2, sharex=ax1)
    ax3 = plt.subplot(6, 1, 3, sharex=ax1)
    ax4 = plt.subplot(6, 1, 4, sharex=ax1)
    ax5 = plt.subplot(6, 1, 5, sharex=ax1)
    ax6 = plt.subplot(6, 1, 6) # Independent x-axis

    ax1.plot(self.waveform)
    ax1.set_title("Original Waveform")

    ax2.plot(self.filtered)
    ax2.set_title("Bandpass Filtered Waveform")

    ax3.plot(self.rectified)
    ax3.set_title("Rectified Waveform")

    ax4.plot(self.envelope)
    ax4.set_title("Amplitude Envelope")

    ax5.plot(self.derivative)
    for beat in self.beats:
        ax5.axvline(beat, color='red', linestyle='--', alpha=0.7,
                    label="Detected Beat" if beat == self.beats[0] else "")

    if ground_truth is not None:
        self.plot_ground_truth(ground_truth, ax5)
        ax5.legend(fontsize='small')
    else:
        ax5.legend()
    ax5.set_title("Derivative of the envelope")

    ax6.plot(self.envSpecFreq, self.spectrum)
    ax6.set_title('Spectrum of the amplitude envelope')
    #bin_size = self.sr / self.N
    #tick_freqs = np.arange(0, 15, bin_size)
    #ax6.set_xticks(tick_freqs)
    ax6.set_xlim(0, 15)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


  def plot_feats(self, out_file):
    print(np.array(self.feats).shape)
    img = librosa.display.specshow(np.array(self.feats))
    plt.savefig(out_file)
  
  def write_wav(self, out_file):
    '''
    Writes the bandpass filtered audio to disk. Must be called after detect_beats().
    '''
    sf.write(out_file, self.filtered, self.sr)

  def save_textgrid(self, timestamps, out_file, xmin = 0):
    '''
    Writes a Praat .TextGrid file with one point tier containing beat annotations
    at the detected timestamps.
    '''
    tmax_s = self.tmax_s
    points = [(t, "Beat") for t in timestamps]
    point_tier = textgrid.PointTier("Beats", points, xmin, tmax_s)
    tg = textgrid.Textgrid(xmin, tmax_s)
    tg.addTier(point_tier)
    tg.save(out_file, format = "long_textgrid", includeBlankSpaces = True)
    print(f"TextGrid saved to {out_file}")