from pathlib import Path
from scipy import signal
import librosa
from scipy.signal import windows
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from praatio import textgrid
from sklearn.preprocessing import MaxAbsScaler
from feat_extractor import MFCC
from bark_env import env3

class RTM():
  def __init__(self, sr, bandpass_left, bandpass_right, env_method, bandpass_order = 1, lowpass_order = 2, lowpass_cutoff = 10, min_dur = 0.05, min_peak_prominence = 0.15, mfcc = False):
    self.bandpass_order = bandpass_order
    self.bandpass_left = bandpass_left
    self.bandpass_right = bandpass_right
    self.lowpass_order = lowpass_order
    self.lowpass_cutoff = lowpass_cutoff
    self.min_dur = min_dur
    self.min_peak_prominence = min_peak_prominence
    self.sr = sr
    self.nyquist = sr / 2
    self.env_method = env_method
    self.feat_extractor = MFCC(sr = self.sr, n_mfcc = 13, frame_length = 512) #TODO: (?) adjust frame length
    self.compute_mfcc = mfcc

  def bandpass(self):
    '''
    First filter. Used to eliminate fricative noise and F0 energy,
    leaving energy in the formants region intact.
    '''
    bandpass_right = self.bandpass_right / self.nyquist
    bandpass_left = self.bandpass_left / self.nyquist
    b, a = signal.butter(self.bandpass_order, [bandpass_left, bandpass_right], btype = 'bandpass')
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
    if self.env_method == 'bark':
      sr = 1000.0
    elif self.env_method == 'vocallic_energy':
      sr = self.sr
    cutoff = self.lowpass_cutoff / (sr / 2)
    b, a = signal.butter(self.lowpass_order, cutoff, btype = 'lowpass')

    return (b, a)
  
  def window_normalize(self, envelope):
      # Windowing (prevent spectral leakage)
      # Subtract mean from entire envelope to remove DC component in the spectrum, then normalize to unit variance
      N = len(envelope)
      tukey = windows.tukey(N, alpha=0.05)
      envelope = envelope * tukey
      envelope = (envelope - np.mean(envelope)) / np.std(envelope)
      
      return envelope

  def envelope_spectrum(self, envelope, max_freq = 10):
    if self.env_method == 'bark':
      sr = 1000.0
    elif self.env_method == 'vocallic_energy':
      sr = self.sr
    envSpec = np.fft.fft(envelope)
    N = len(envSpec)
    freq_axis = np.fft.fftfreq(N, 1 / sr)[:N // 2]
    envSpec = abs(envSpec[:N // 2]) / (N // 2) # Divided by N for numerical stability

    freq_mask = freq_axis <= max_freq
    envSpec = envSpec[freq_mask]
    freq_axis = freq_axis[freq_mask]

    return envSpec, freq_axis

  def process(self,
              waveform,
              identifier,
              write_dir,            
              gt_tg_dir,
              write_tgs = False,
              write_filtered = False,
              write_plots = False):
    '''
    Core function. Performs all signal processing steps.
    '''
    # Waveform and intermediate signals are stored as instance attributes
    tmax_s = len(waveform) / self.sr
    scaler = MaxAbsScaler()

    if self.env_method == 'bark':
      envelope = env3(waveform, self.sr, fs_out = 1000.0)
      envelope = signal.filtfilt(*self.lowpass(), envelope)
      envelope = self.window_normalize(envelope)

    elif self.env_method == 'vocallic_energy':
      filtered = signal.lfilter(*self.bandpass(), waveform)
      rectified = self.rectify(filtered)
      envelope = signal.lfilter(*self.lowpass(), rectified)

      # Moving average smoothing (50ms window)
      envelope = pd.Series(envelope).rolling(int(self.sr/100), center=False, min_periods=1).mean().to_numpy()

      envelope = self.window_normalize(envelope)

    derivative = np.gradient(envelope)
    derivative = scaler.fit_transform(derivative.reshape(-1, 1)).reshape(-1)
    peaks, _ = signal.find_peaks(derivative,
                                 #distance = self.sr * self.min_dur,
                                 prominence = self.min_peak_prominence)

    #Detect beats
    # Note: self.beats will store the frame indices of the detected beats
    beats = []
    feats = []
    beat_intervals = []

    for index in peaks:
      peak_index = index
      if derivative[peak_index] > 0:
        beats.append(peak_index)

        if self.compute_mfcc:
          mfcc_feats = self.feat_extractor.extract_mfcc(waveform, identifier, peak_index)
          feats.append(mfcc_feats[:, 0]) # len(beat_frames), n_mfcc

      else:
        continue

    for idx, _ in enumerate(beats):
      if idx != 0:
        interval = (beats[idx] - beats[idx - 1]) / self.sr
        beat_intervals.append(interval)
      else:
        beat_intervals.append(0)

    # Extract envelope features
      # Spectrum of the magnitude envelope
    assert not np.any(np.isnan(envelope)), "Envelope contains NaN values"
    assert not np.any(np.isinf(envelope)), "Envelope contains infinite values"
    max_freq = 10 # Hz
    if self.env_method == 'bark':
      out_sr = 1000
    elif self.env_method == 'vocallic_energy':
      out_sr = self.sr

    if len(envelope) <= 10*self.sr: # Assuming max audio len is 10 seconds
      envelope_padded = np.pad(envelope, (0, 10*out_sr - len(envelope)))
      spectrum, envSpecFreq = self.envelope_spectrum(envelope_padded)

    out = {'beat_frames': beats,
           'envelope': envelope,
           'envelope_spectrum':spectrum[1:], # Remove the 0 component
           'spectrum_freq_bins':envSpecFreq[1:], # Remove the 0 component
           'beat_intervals':beat_intervals}

    if self.compute_mfcc:
      out['mfcc_features'] = feats

    if write_tgs:
      tg_path = Path(write_dir) / 'textgrids'
      tg_path.mkdir(parents=True, exist_ok=True)
      out_file = tg_path / Path(identifier).with_suffix('.TextGrid')
      self.save_textgrid(beats, out_file, tmax_s)

    if write_filtered:
      filtered_path = Path(write_dir) / 'filtered_wavs'
      filtered_path.mkdir(parents=True, exist_ok=True)
      out_file = filtered_path / identifier
      self.write_wav(out_file, filtered)

    if write_plots:
      gt_path = Path(gt_tg_dir) / Path(identifier).with_suffix('.TextGrid')
      plots_path = Path(write_dir) / 'plots'
      plots_path.mkdir(parents=True, exist_ok=True)
      out_file = plots_path / Path(identifier).with_suffix('.png')
      if gt_path is not None and gt_path.is_file():
        self.plot_filtered(waveform, envelope, derivative, beats, out_file, envSpecFreq, spectrum, ground_truth = str(gt_path))
      else:
        self.plot_filtered(waveform, envelope, derivative, beats, out_file, envSpecFreq, spectrum)


    return out

  def plot_ground_truth(self, ground_truth, plot):
      tg = textgrid.openTextgrid(ground_truth, includeEmptyIntervals=True)
      gt_tier = tg.getTier('beats')
      gt_timestamps = [entry.time * self.sr for entry in gt_tier.entries]
      for i, beat in enumerate(gt_timestamps):
        label = "Ground truth" if i == 0 else ""
        plot.axvline(beat, color='red', linestyle='-', alpha=0.7, label = label)

  def plot_filtered(self,
                    waveform,
                    envelope,
                    derivative,
                    beats,
                    out_file,
                    envSpecFreq,
                    spectrum,
                    ground_truth = None):

    fig = plt.figure(figsize=(12, 8))

    # Create 6 subplots, with the first 5 sharing x-axis
    ax1 = plt.subplot(4, 1, 1)
    ax2 = plt.subplot(4, 1, 2)
    ax3 = plt.subplot(4, 1, 3, sharex=ax2)
    ax4 = plt.subplot(4, 1, 4)

    ax1.plot(waveform)
    ax1.set_title("Original Waveform")

    ax2.plot(envelope)
    ax2.set_title("Amplitude Envelope")

    ax3.plot(derivative)
    for beat in beats:
        ax3.axvline(beat, color='red', linestyle='--', alpha=0.7,
                    label="Detected Beat" if beat == beats[0] else "")

    if ground_truth is not None:
        self.plot_ground_truth(ground_truth, ax3)
        ax3.legend(fontsize='small')
    else:
        ax3.legend()
    ax3.set_title("Derivative of the envelope")

    ax4.plot(envSpecFreq, spectrum, linestyle='--', marker='o', markersize=4)
    ax4.set_title('Spectrum of the amplitude envelope')
    ax4.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()

    # ax2.plot(self.filtered)
    # ax2.set_title("Bandpass Filtered Waveform")

    # ax3.plot(self.rectified)
    # ax3.set_title("Rectified Waveform")

  def plot_feats(self, out_file):
    img = librosa.display.specshow(np.array(self.feats))
    plt.savefig(out_file)

  def plot_wt(self, out_file):
    plt.imshow(np.abs(self.Wx), aspect='auto', extent=[0, 1, self.freqs_cwt[-1], self.freqs_cwt[0]], cmap='viridis')
    plt.savefig(out_file)

  def write_wav(self, out_file, wav):
    '''
    Writes the bandpass filtered audio to disk. Must be called after detect_beats().
    '''
    sf.write(out_file, wav, self.sr)

  def save_textgrid(self, timestamps, out_file, tmax_s, xmin = 0):
    '''
    Writes a Praat .TextGrid file with one point tier containing beat annotations
    at the detected timestamps.
    '''
    points = [(t, "Beat") for t in timestamps]
    point_tier = textgrid.PointTier("Beats", points, xmin, tmax_s)
    tg = textgrid.Textgrid(xmin, tmax_s)
    tg.addTier(point_tier)
    tg.save(out_file, format = "long_textgrid", includeBlankSpaces = True)