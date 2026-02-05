from pathlib import Path
from scipy import signal
import librosa
from scipy.signal import windows, lfilter, filtfilt
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from praatio import textgrid
from sklearn.preprocessing import MaxAbsScaler
from feat_extractor import MFCC
from abc import ABC, abstractmethod


class EnvelopeExtractor(ABC):
  """Base class for envelope extraction methods."""
  
  @abstractmethod
  def extract(self, waveform: np.ndarray) -> np.ndarray:
    """Extract amplitude envelope from waveform."""
    pass
  
  @property
  @abstractmethod
  def output_sr(self) -> float:
    """Return the output sampling rate of the envelope."""
    pass


class BarkEnvelopeExtractor(EnvelopeExtractor):
  """
  Envelope extraction using Bark-scale filterbank.
  Python port of MATLAB env3.m from Oganian & Chang (2019) / Schotola (1984) method.
  """
  
  def __init__(self, sr: float, fs_out: float = 1000.0):
    self.sr = sr
    self.fs_out = fs_out
  
  @property
  def output_sr(self) -> float:
    return self.fs_out
  
  def extract(self, waveform: np.ndarray) -> np.ndarray:
    """
    Extract envelope using Bark-scale filterbank.
    
    Parameters
    ----------
    waveform : np.ndarray
        1-D audio waveform.
    
    Returns
    -------
    envelope : np.ndarray
        1-D loudness envelope sampled at fs_out.
    """
    x = np.asarray(waveform, dtype=np.float64).reshape(-1)
    n = x.size
    t_n = np.arange(n) / self.sr
    t_step = 1.0 / self.sr

    sr = self.fs_out
    n1 = int(np.floor(n * sr / self.sr))
    t1 = np.arange(n1) / sr

    # FFT of input
    X = np.fft.fft(x, n=n)

    # Positive-frequency bins (rfft-like indexing)
    frqs = np.fft.rfftfreq(n, d=1.0 / self.sr)  # length nfrqs = n//2 + 1
    nfrqs = frqs.size

    # Bark scale (Zwicker & Terhardt 1980), then drop DC and Nyquist
    z = 13.0 * np.arctan(0.76 * frqs / 1000.0) + 3.5 * (np.arctan(frqs / 7500.0) ** 2)
    z = z[1:-1]

    # RC smoothing with 1.3 ms time constant
    tau = 0.0013
    r = np.exp(-t_step / tau)
    b1 = np.array([1.0 - r], dtype=np.float64)
    a1 = np.array([1.0, -r], dtype=np.float64)

    # Allocate
    czs = np.arange(1, 23)  # 1..22
    nczs = czs.size
    Nv = np.zeros((n1, nczs), dtype=np.float64)
    F = np.zeros(n, dtype=np.float64)  # frequency-domain weighting

    # forward then reverse IIR filtering
    def zero_phase_rc(y: np.ndarray) -> np.ndarray:
      y_fwd = lfilter(b1, a1, y)
      y_bwd = np.flip(lfilter(b1, a1, np.flip(y_fwd)))
      return y_bwd

    eps = 1e-12  # to avoid log(0)

    for i, cz in enumerate(czs):
      # Build filter weights for positive freqs excluding DC+Nyquist
      w = 10.0 ** (
          0.7
          - 0.75 * ((z - cz) - 0.215)
          - 1.75 * (0.196 + ((z - cz) - 0.215) ** 2)
      )

      # Place into full-length F
      F.fill(0.0)
      F[1 : nfrqs - 1] = w  # indices align with z (bins 1..nfrqs-2)

      # Apply frequency weighting and go back to time domain
      lev = np.real(np.fft.ifft(X * F, n=n))

      # Square-law rectification
      lev = lev ** 2

      # 1.3 ms smoothing, zero-phase (forward+reverse)
      lev = zero_phase_rc(lev)

      # Log nonlinearity -> excitation level, then "delog" of 0.5*Lev -> sqrt in linear domain
      lev = np.log(lev + eps)
      specific = np.exp(0.5 * lev)

      # Resample to fs_out
      Nv[:, i] = np.interp(t1, t_n, specific)

    Nv = np.maximum(0.0, Nv)

    # Weighted sum across bands: gv(czs<3)=0; gv(czs>19)=-1; else +1
    gv = np.ones(nczs, dtype=np.float64)
    gv[czs < 3] = 0.0
    gv[czs > 19] = -1.0

    Nm = Nv @ gv

    # Repeated 3-point triangular/box smoothing with filtfilt, n=13
    b = np.ones(3, dtype=np.float64) / 3.0
    for _ in range(13):
      Nm = filtfilt(b, [1.0], Nm)

    return Nm


class VocallicEnergyExtractor(EnvelopeExtractor):
  """
  Envelope extraction using bandpass filtering and rectification.
  Eliminates fricative noise and F0 energy, leaving formant energy intact.
  """
  
  def __init__(self, sr: float, bandpass_left: float = 700, bandpass_right: float = 1300,
               bandpass_order: int = 1, lowpass_order: int = 2, lowpass_cutoff: float = 10):
    self.sr = sr
    self.bandpass_left = bandpass_left
    self.bandpass_right = bandpass_right
    self.bandpass_order = bandpass_order
    self.lowpass_order = lowpass_order
    self.lowpass_cutoff = lowpass_cutoff
    self.nyquist = sr / 2
  
  @property
  def output_sr(self) -> float:
    return self.sr
  
  def _bandpass(self):
    """
    First filter. Used to eliminate fricative noise and F0 energy,
    leaving energy in the formants region intact.
    """
    bandpass_right = self.bandpass_right / self.nyquist
    bandpass_left = self.bandpass_left / self.nyquist
    b, a = signal.butter(self.bandpass_order, [bandpass_left, bandpass_right], btype='bandpass')
    return (b, a)
  
  def _rectify(self, waveform: np.ndarray) -> np.ndarray:
    """Signal rectification using absolute values."""
    return np.abs(waveform)
  
  def _lowpass(self):
    """Second filter. Used to produce a smooth amplitude envelope."""
    cutoff = self.lowpass_cutoff / (self.sr / 2)
    b, a = signal.butter(self.lowpass_order, cutoff, btype='lowpass')
    return (b, a)
  
  def extract(self, waveform: np.ndarray) -> np.ndarray:
    """
    Extract amplitude envelope using bandpass filtering and rectification.
    
    Parameters
    ----------
    waveform : np.ndarray
        1-D audio waveform.
    
    Returns
    -------
    envelope : np.ndarray
        1-D amplitude envelope at original sampling rate.
    """
    filtered = signal.lfilter(*self._bandpass(), waveform)
    rectified = self._rectify(filtered)
    envelope = signal.lfilter(*self._lowpass(), rectified)
    
    # Moving average smoothing (50ms window)
    envelope = pd.Series(envelope).rolling(int(self.sr / 100), center=False, min_periods=1).mean().to_numpy()
    
    return envelope


class RTM():
  def __init__(self, sr, bandpass_left, bandpass_right, env_method, bandpass_order=1, lowpass_order=2, lowpass_cutoff=10, min_dur=0.05, min_peak_prominence=0.15, mfcc=False):
    self.min_dur = min_dur
    self.min_peak_prominence = min_peak_prominence
    self.sr = sr
    self.env_method = env_method
    self.feat_extractor = MFCC(sr=self.sr, n_mfcc=13, frame_length=512)
    self.compute_mfcc = mfcc
    
    # Initialize the appropriate envelope extractor
    if env_method == 'bark':
      self.envelope_extractor = BarkEnvelopeExtractor(sr=sr, fs_out=1000.0)
    elif env_method == 'vocallic_energy':
      self.envelope_extractor = VocallicEnergyExtractor(
        sr=sr,
        bandpass_left=bandpass_left,
        bandpass_right=bandpass_right,
        bandpass_order=bandpass_order,
        lowpass_order=lowpass_order,
        lowpass_cutoff=lowpass_cutoff
      )
    else:
      raise ValueError(f"Unknown envelope method: {env_method}. Choose 'bark' or 'vocallic_energy'.")
    
    # Store lowpass params for envelope spectrum (used in both methods after extraction)
    self.lowpass_order = lowpass_order
    self.lowpass_cutoff = lowpass_cutoff

  def _lowpass_for_bark(self):
    """Lowpass filter for post-processing bark envelope."""
    sr = 1000.0
    cutoff = self.lowpass_cutoff / (sr / 2)
    b, a = signal.butter(self.lowpass_order, cutoff, btype='lowpass')
    return (b, a)

  def window_normalize(self, envelope):
    # Windowing (prevent spectral leakage)
    # Subtract mean from entire envelope to remove DC component in the spectrum, then normalize to unit variance
    N = len(envelope)
    tukey = windows.tukey(N, alpha=0.05)
    envelope = envelope * tukey
    envelope = (envelope - np.mean(envelope)) / np.std(envelope)
    
    return envelope

  def envelope_spectrum(self, envelope, max_freq=10):
    sr = self.envelope_extractor.output_sr
    envSpec = np.fft.fft(envelope)
    N = len(envSpec)
    freq_axis = np.fft.fftfreq(N, 1 / sr)[:N // 2]
    envSpec = abs(envSpec[:N // 2]) / (N // 2)  # Divided by N for numerical stability

    freq_mask = freq_axis <= max_freq
    envSpec = envSpec[freq_mask]
    freq_axis = freq_axis[freq_mask]

    return envSpec, freq_axis

  def process(self,
              waveform,
              identifier,
              write_dir,            
              gt_tg_dir,
              write_tgs=False,
              write_filtered=False,
              write_plots=False):
    '''
    Core function. Performs all signal processing steps.
    '''
    tmax_s = len(waveform) / self.sr
    scaler = MaxAbsScaler()

    # Extract envelope using the configured extractor
    envelope = self.envelope_extractor.extract(waveform)
    
    # Post-processing for bark method
    if self.env_method == 'bark':
      envelope = signal.filtfilt(*self._lowpass_for_bark(), envelope)
    
    envelope = self.window_normalize(envelope)

    derivative = np.gradient(envelope)
    derivative = scaler.fit_transform(derivative.reshape(-1, 1)).reshape(-1)
    peaks, _ = signal.find_peaks(derivative,
                                 prominence=self.min_peak_prominence)

    # Detect beats
    beats = []
    feats = []
    beat_intervals = []

    for index in peaks:
      peak_index = index
      if derivative[peak_index] > 0:
        beats.append(peak_index)

        if self.compute_mfcc:
          mfcc_feats = self.feat_extractor.extract_mfcc(waveform, identifier, peak_index)
          feats.append(mfcc_feats[:, 0])

      else:
        continue

    for idx, _ in enumerate(beats):
      if idx != 0:
        interval = (beats[idx] - beats[idx - 1]) / self.sr
        beat_intervals.append(interval)
      else:
        beat_intervals.append(0)

    # Extract envelope features
    assert not np.any(np.isnan(envelope)), "Envelope contains NaN values"
    assert not np.any(np.isinf(envelope)), "Envelope contains infinite values"
    max_freq = 10  # Hz
    out_sr = self.envelope_extractor.output_sr

    if len(envelope) <= 10 * out_sr:  # Assuming max audio len is 10 seconds
      envelope_padded = np.pad(envelope, (0, int(10 * out_sr) - len(envelope)))
      spectrum, envSpecFreq = self.envelope_spectrum(envelope_padded)

    out = {'beat_frames': beats,
           'envelope': envelope,
           'envelope_spectrum': spectrum[1:],
           'spectrum_freq_bins': envSpecFreq[1:],
           'beat_intervals': beat_intervals}

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
        self.plot_filtered(waveform, envelope, derivative, beats, out_file, envSpecFreq, spectrum, ground_truth=str(gt_path))
      else:
        self.plot_filtered(waveform, envelope, derivative, beats, out_file, envSpecFreq, spectrum)

    return out

  def plot_ground_truth(self, ground_truth, plot):
    tg = textgrid.openTextgrid(ground_truth, includeEmptyIntervals=True)
    gt_tier = tg.getTier('beats')
    gt_timestamps = [entry.time * self.sr for entry in gt_tier.entries]
    for i, beat in enumerate(gt_timestamps):
      label = "Ground truth" if i == 0 else ""
      plot.axvline(beat, color='red', linestyle='-', alpha=0.7, label=label)

  def plot_filtered(self,
                    waveform,
                    envelope,
                    derivative,
                    beats,
                    out_file,
                    envSpecFreq,
                    spectrum,
                    ground_truth=None):

    fig = plt.figure(figsize=(12, 8))

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

  def write_wav(self, out_file, wav):
    '''
    Writes the bandpass filtered audio to disk. Must be called after detect_beats().
    '''
    sf.write(out_file, wav, self.sr)

  def save_textgrid(self, timestamps, out_file, tmax_s, xmin=0):
    '''
    Writes a Praat .TextGrid file with one point tier containing beat annotations
    at the detected timestamps.
    '''
    points = [(t, "Beat") for t in timestamps]
    point_tier = textgrid.PointTier("Beats", points, xmin, tmax_s)
    tg = textgrid.Textgrid(xmin, tmax_s)
    tg.addTier(point_tier)
    tg.save(out_file, format="long_textgrid", includeBlankSpaces=True)