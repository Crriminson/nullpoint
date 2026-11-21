"""Nystagmus metrics from an eye-position trace (degrees) during one gaze hold.

All functions take x (deg) sampled at fs (Hz). Methods:
  frequency   autocorrelation of the band-passed trace (robust for sawtooth/jerk waveforms whose
              spectrum is a harmonic comb; the FFT peak is reported alongside)
  amplitude   median per-cycle peak-to-peak, cycles cut at the detected period
  velocity    Savitzky-Golay derivative
  fast phases |v| above an adaptive threshold -> direction (sign of fast-phase velocity)
  waveform    heuristic baseline: velocity skewness + fast-phase fraction (a 1D-CNN replaces this)
  foveation   samples with |v| < 4 deg/s (Dell'Osso-style velocity criterion); gaze estimate =
              median position over foveation samples, which is NOT the trace mean for jerk waveforms
  intensity   amplitude x frequency (deg/s), the conventional severity number
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import butter, sosfiltfilt, savgol_filter, welch, find_peaks

FOV_VEL = 4.0  # deg/s


def bandpass(x, fs, lo=0.7, hi=None):
    hi = hi or min(25.0, 0.45 * fs)
    sos = butter(2, [lo, hi], "bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def velocity(x, fs):
    w = max(5, int(round(0.03 * fs)) | 1)        # ~30 ms window
    return savgol_filter(x, w, 2, deriv=1, delta=1 / fs)


def frequency(x, fs, fmin=1.0, fmax=8.0):
    """(f_autocorr, f_welch, periodicity 0..1)."""
    y = bandpass(x, fs, lo=0.5 * fmin)
    y = y - y.mean()
    ac = np.correlate(y, y, "full")[len(y) - 1:]
    ac /= ac[0] + 1e-12
    lo, hi = int(fs / fmax), min(int(fs / fmin), len(ac) - 1)
    pk, props = find_peaks(ac[lo:hi], height=0.05)
    if len(pk):
        # first strong peak (avoid picking a multiple of the period)
        best = pk[np.argmax(props["peak_heights"])]
        strong = pk[props["peak_heights"] > 0.8 * props["peak_heights"].max()]
        k = strong[0] + lo
        # parabolic refinement
        if 0 < k < len(ac) - 1:
            a, b, c = ac[k - 1:k + 2]; den = a - 2 * b + c
            k = k + (0.5 * (a - c) / den if den else 0)
        f_ac, per = fs / k, float(ac[int(round(k))])
    else:
        f_ac, per = np.nan, 0.0
    f, P = welch(y, fs=fs, nperseg=min(len(y), int(4 * fs)))
    band = (f >= fmin) & (f <= fmax)
    f_w = float(f[band][np.argmax(P[band])]) if band.any() else np.nan
    return float(f_ac), f_w, per


def amplitude(x, fs, f):
    """Median per-cycle peak-to-peak (deg) of the band-passed trace."""
    if not np.isfinite(f) or f <= 0:
        return float(np.percentile(x, 95) - np.percentile(x, 5))
    y = bandpass(x, fs, lo=max(0.3, 0.3 * f))
    n = int(round(fs / f))
    pp = [np.ptp(y[i:i + n]) for i in range(0, len(y) - n + 1, n)]
    return float(np.median(pp)) if pp else float(np.ptp(y))


@dataclass
class HoldMetrics:
    freq_hz: float
    freq_welch_hz: float
    periodicity: float
    amp_deg: float
    intensity: float
    fast_dir: str            # "left" / "right" / "none"
    waveform_heuristic: str  # none / pendular / jerk_l / jerk_r
    vel_skew: float
    fast_frac: float
    foveation_frac: float
    gaze_mean_deg: float
    gaze_foveation_deg: float
    n: int

    def dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def analyse(x: np.ndarray, fs: float) -> HoldMetrics:
    x = np.asarray(x, float)
    good = np.isfinite(x)
    if good.sum() < len(x):
        x = x.copy(); x[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), x[good])
    f_ac, f_w, per = frequency(x, fs)
    amp = amplitude(x, fs, f_ac if np.isfinite(f_ac) else f_w)
    v = velocity(x, fs)
    vb = velocity(bandpass(x, fs, lo=0.5), fs)
    mad = np.median(np.abs(vb - np.median(vb))) * 1.4826
    thr = max(20.0, 4 * mad)
    fast = np.abs(vb) > thr
    fast_frac = float(fast.mean())
    # direction: the fast phase is several times faster than even the (accelerating) end of the
    # slow phase, so compare the velocity extremes in each direction
    hi, lo = np.percentile(vb, 99.5), -np.percentile(vb, 0.5)
    if max(hi, lo) > thr and max(hi, lo) > 1.6 * min(hi, lo):
        fast_dir = "right" if hi > lo else "left"
    else:
        fast_dir = "none"
    from scipy.stats import skew
    sk = float(skew(vb))
    # heuristic waveform baseline
    if per < 0.25 or amp < 0.4:
        wf = "none"
    elif abs(sk) > 0.8 and fast_dir != "none":
        wf = "jerk_r" if fast_dir == "right" else "jerk_l"
    else:
        wf = "pendular"
    fov = np.abs(v) < FOV_VEL
    gaze_fov = float(np.median(x[fov])) if fov.sum() > 5 else float(np.median(x))
    return HoldMetrics(f_ac, f_w, per, amp, amp * (f_ac if np.isfinite(f_ac) else f_w), fast_dir, wf, sk,
                       fast_frac, float(fov.mean()), float(np.mean(x)), gaze_fov, len(x))
