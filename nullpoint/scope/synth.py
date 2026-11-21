"""Synthetic nystagmus generator with ground truth.

Used to (1) validate the measurement pipeline (known frequency/amplitude/waveform/foveation),
(2) train the waveform classifier, (3) inject realistic nystagmus into recorded webcam
streams for the replay benchmark.

Waveform families (after Dell'Osso & Daroff's INS waveform taxonomy, simplified):
  * "jerk_r" / "jerk_l": slow phase drifting away from target with increasing velocity
    (exponential ramp, typical of INS), fast corrective saccade back. Direction named after
    the FAST phase, as is clinical convention.
  * "pendular": quasi-sinusoidal, optionally with a small harmonic.
  * "none": fixation with only microsaccade-like noise and drift.
Every beat includes a foveation period (eye nearly still near the target) whose duration is
a parameter: this is what a good gaze estimator should find.
Frequency and amplitude wobble slowly within a trace (real nystagmus is not stationary).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

CLASSES = ["none", "pendular", "jerk_l", "jerk_r"]


@dataclass
class NystParams:
    kind: str = "jerk_r"
    freq_hz: float = 3.0
    amp_deg: float = 4.0          # peak-to-peak
    foveation_frac: float = 0.15  # fraction of each cycle spent in foveation
    jitter: float = 0.08          # cycle-to-cycle variability of period and amplitude
    harmonic: float = 0.15        # pendular only: relative 2nd harmonic


def _beat(kind: str, n: int, amp: float, fov_frac: float, rng, harmonic=0.15) -> np.ndarray:
    """One cycle of n samples, starting at the target (0), deviation in degrees."""
    u = np.linspace(0, 1, n, endpoint=False)
    nf = max(1, int(round(fov_frac * n)))
    if kind == "pendular":
        ph = 2 * np.pi * u
        w = np.sin(ph) + harmonic * np.sin(2 * ph + rng.uniform(0, np.pi))
        w = w / (np.ptp(w) + 1e-9) * amp
        # foveation: flatten around the zero crossing at the start
        k = np.arange(n)
        w = np.where(k < nf, w[nf] * (k / nf) ** 3, w)
        return w - w[:nf].mean() if nf > 1 else w
    if kind in ("jerk_l", "jerk_r"):
        n_fast = max(2, int(round(0.06 * n)))          # fast phase ~6 % of the cycle
        n_slow = n - nf - n_fast
        s = np.linspace(0, 1, max(n_slow, 2))
        tau = rng.uniform(0.25, 0.45)
        slow = (np.exp(s / tau) - 1) / (np.exp(1 / tau) - 1)  # increasing-velocity ramp 0 -> 1
        fast = 1 - (1 - np.cos(np.linspace(0, np.pi, n_fast))) / 2  # 1 -> 0, saccade-like
        w = np.r_[np.zeros(nf) + rng.normal(0, 0.01, nf), slow, fast][:n] * amp
        # slow phase drifts AWAY from fast-phase direction: jerk_r = fast phase rightward
        return -w if kind == "jerk_r" else w
    return np.zeros(n)


def generate(p: NystParams, duration: float, fs: float, seed: int = 0, noise_deg: float = 0.0,
             drift_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (t, x_deg, foveating_mask). x is eye position relative to the intended target."""
    rng = np.random.default_rng(seed)
    N = int(duration * fs)
    x = np.zeros(N)
    fov = np.zeros(N, bool)
    i = 0
    slow = rng.uniform(0, 2 * np.pi)
    while i < N:
        mod = 1 + 0.5 * p.jitter * np.sin(slow + i / fs * 0.4)
        f = p.freq_hz * mod * (1 + rng.normal(0, p.jitter * 0.5))
        a = p.amp_deg * (1 + rng.normal(0, p.jitter))
        n = max(4, int(round(fs / max(f, 0.2))))
        if p.kind == "none":
            b = np.zeros(n)
            nf = n
        else:
            b = _beat(p.kind, n, max(a, 0.0), p.foveation_frac, rng, p.harmonic)
            nf = max(1, int(round(p.foveation_frac * n)))
        m = min(n, N - i)
        x[i:i + m] = b[:m]
        fov[i:i + min(nf, m)] = True
        i += n
    t = np.arange(N) / fs
    if drift_deg:
        x += drift_deg * np.sin(2 * np.pi * 0.15 * t + rng.uniform(0, 6))
    if p.kind == "none":
        # occasional microsaccades
        for k in rng.choice(N, size=max(1, int(duration * 1.5)), replace=False):
            x[k:] += rng.normal(0, 0.25)
        x -= np.median(x)
    if noise_deg:
        x += rng.normal(0, noise_deg, N)
    return t, x, fov


def random_params(rng, kind: str | None = None) -> NystParams:
    """Clinically plausible INS ranges (freq ~1.5-6 Hz, amp 0.5-15 deg p-p)."""
    kind = kind or rng.choice(CLASSES)
    return NystParams(kind=kind, freq_hz=float(rng.uniform(1.2, 6.0)), amp_deg=float(np.exp(rng.uniform(np.log(0.5), np.log(15)))),
                      foveation_frac=float(rng.uniform(0.05, 0.35)), jitter=float(rng.uniform(0.02, 0.15)),
                      harmonic=float(rng.uniform(0, 0.3)))


def dataset(n: int, fs: float, win_s: float, seed: int = 0, noise_range=(0.05, 1.0)):
    """Windows for the waveform classifier: X (n, win) in deg, y (n,), params list."""
    rng = np.random.default_rng(seed)
    W = int(win_s * fs)
    X = np.zeros((n, W), np.float32); y = np.zeros(n, np.int64); ps = []
    for i in range(n):
        p = random_params(rng)
        _, x, _ = generate(p, win_s + 1.0, fs, seed=int(rng.integers(1 << 31)),
                           noise_deg=float(rng.uniform(*noise_range)) * (1 if p.kind != "none" else 1.5),
                           drift_deg=float(rng.uniform(0, 2)))
        st = int(rng.integers(0, len(x) - W))
        X[i] = x[st:st + W]; y[i] = CLASSES.index(p.kind); ps.append(asdict(p))
    return X, y, ps
