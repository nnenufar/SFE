import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import lfilter, filtfilt
import os
import matplotlib.pyplot as plt

WAV_PATH = "/hadatasets/joao.lima/data/quick_samples/audios/0016_001401_Su.wav"
OUT_PATH = "/hadatasets/joao.lima/data/quick_samples/out"
FS_OUT = 1000
DO_SMOOTH = True


def env3(signal_in: np.ndarray, fs: float, fs_out: float = 1000.0) -> np.ndarray:
    """
    Python port of MATLAB env3.m from Oganian & Chang (2019) / Schotola (1984) method.

    Parameters
    ----------
    signal_in : np.ndarray
        1-D audio waveform.
    fs : float
        Sampling rate of signal_in (Hz).
    fs_out : float, optional
        Output sampling rate for envelope (Hz). Default 1000.

    Returns
    -------
    envelope_out : np.ndarray
        1-D loudness-like envelope sampled at fs_out.
    """
    x = np.asarray(signal_in, dtype=np.float64).reshape(-1)
    n = x.size
    t_n = np.arange(n) / fs
    t_step = 1.0 / fs

    sr = fs_out
    n1 = int(np.floor(n * sr / fs))
    t1 = np.arange(n1) / sr

    # FFT of input
    X = np.fft.fft(x, n=n)

    # Positive-frequency bins (rfft-like indexing)
    frqs = np.fft.rfftfreq(n, d=1.0 / fs)  # length nfrqs = n//2 + 1
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