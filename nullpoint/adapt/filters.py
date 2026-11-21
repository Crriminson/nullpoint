"""Online filters that remove involuntary oscillation from a gaze signal.

All filters are causal (sample-by-sample) so they can run live at 30 fps.

* OneEuro      : standard adaptive low-pass for jittery pointers (baseline; Casiez et al. 2012)
* PeriodMean   : moving average over exactly k nystagmus periods (a periodic component of
                 period T has zero mean over any window of length kT, so it cancels exactly)
* FLC          : Fourier linear combiner (Riviere & Thakor 1996 family): LMS-learned harmonic model
                 of the oscillation at the MEASURED frequency (re-checked by autocorrelation),
                 subtracted from the signal. WFLC (gradient frequency adaptation) is kept for
                 reference but is unreliable at 30 fps, see class docstring
* FoveationMean: velocity-weighted average: samples taken while the eye is slow (foveation-like)
                 get full weight, fast-phase samples little weight. Fixes the jerk-waveform bias
                 of plain averaging (see docs: 0.73 deg -> 0.18 deg on synthetic jerk traces)
* AdaptChain   : the configuration Nullpoint uses, built from a nystagmus profile
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


class OneEuro:
    def __init__(self, fs=30.0, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.fs, self.mc, self.beta, self.dc = fs, min_cutoff, beta, d_cutoff
        self.x = self.dx = None

    @staticmethod
    def _a(fc, fs):
        tau = 1 / (2 * math.pi * fc); return 1 / (1 + tau * fs)

    def __call__(self, v):
        if self.x is None:
            self.x, self.dx = v, 0.0; return v
        dx = (v - self.x) * self.fs
        self.dx += self._a(self.dc, self.fs) * (dx - self.dx)
        fc = self.mc + self.beta * abs(self.dx)
        self.x += self._a(fc, self.fs) * (v - self.x)
        return self.x


class PeriodMean:
    def __init__(self, fs=30.0, freq=None, k=1, default_s=0.33):
        n = int(round(k * fs / freq)) if freq else int(round(default_s * fs))
        self.buf = deque(maxlen=max(1, n)); self.n = max(1, n)

    def __call__(self, v):
        self.buf.append(v); return float(np.mean(self.buf))


class WFLC:
    """Riviere's WFLC: x_hat = sum_r w_r sin(r*phi) + w_{r+M} cos(r*phi), phi += omega.

    Input should be the oscillatory part (we feed a high-passed signal and subtract the
    estimate from the original)."""

    def __init__(self, fs=30.0, f0=3.0, M=2, mu=0.01, mu0=0.0005, mu_b=0.0, fmin=0.8, fmax=8.0):
        self.fs, self.M, self.mu, self.mu0 = fs, M, mu, mu0
        self.w = np.zeros(2 * M)
        self.omega = 2 * math.pi * f0 / fs
        self.phi = 0.0
        self.wmin, self.wmax = 2 * math.pi * fmin / fs, 2 * math.pi * fmax / fs

    @property
    def freq(self):
        return self.omega * self.fs / (2 * math.pi)

    def __call__(self, s):
        self.phi += self.omega
        r = np.arange(1, self.M + 1)
        x = np.r_[np.sin(r * self.phi), np.cos(r * self.phi)]
        est = float(self.w @ x)
        err = s - est
        # frequency update (gradient wrt omega through phi)
        g = float(np.sum(r * (self.w[:self.M] * np.cos(r * self.phi) - self.w[self.M:] * np.sin(r * self.phi))))
        self.omega = float(np.clip(self.omega + 2 * self.mu0 * err * g, self.wmin, self.wmax))
        self.w += 2 * self.mu * err * x
        return est


class FLC:
    """Fourier linear combiner at a given frequency (adaptive harmonic canceller).

    Weights of M harmonics are learned by LMS; the frequency comes from the measured profile and
    is re-estimated every `retune_s` seconds by autocorrelation of the recent oscillatory signal
    (clamped to +-30 % of the profile frequency). Riviere & Thakor's WFLC adapts the frequency
    by gradient instead; on 30 fps gaze data that gradient is unreliable (the LMS weights absorb
    phase drift), so the frequency is taken from the measurement, which is the point of Scope."""

    def __init__(self, fs=30.0, f0=3.0, M=3, mu=0.03, retune_s=2.0):
        self.fs, self.M, self.mu, self.f0 = fs, M, mu, f0
        self.f = f0
        self.w = np.zeros(2 * M)
        self.phi = 0.0
        self.hist = deque(maxlen=int(4 * fs))
        self.n, self.retune = 0, int(retune_s * fs)

    def __call__(self, s):
        self.phi += 2 * math.pi * self.f / self.fs
        r = np.arange(1, self.M + 1)
        x = np.r_[np.sin(r * self.phi), np.cos(r * self.phi)]
        est = float(self.w @ x)
        err = s - est
        self.w += 2 * self.mu * err * x
        self.hist.append(s); self.n += 1
        if self.n % self.retune == 0 and len(self.hist) == self.hist.maxlen:
            self.f = self._estimate()
        return est

    def _estimate(self):
        y = np.array(self.hist) - np.mean(self.hist)
        ac = np.correlate(y, y, "full")[len(y) - 1:]
        lo, hi = int(self.fs / (1.3 * self.f0)), int(self.fs / (0.7 * self.f0)) + 1
        hi = min(hi, len(ac) - 1)
        if hi <= lo + 1 or ac[0] <= 0:
            return self.f
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] / ac[0] < 0.2:
            return self.f
        return float(self.fs / k)


class HighPass:
    """Causal 1st-order high-pass."""
    def __init__(self, fs=30.0, fc=0.8):
        rc = 1 / (2 * math.pi * fc); self.a = rc / (rc + 1 / fs); self.y = 0.0; self.x = None

    def __call__(self, x):
        if self.x is None:
            self.x = x; return 0.0
        self.y = self.a * (self.y + x - self.x); self.x = x; return self.y


class FoveationMean:
    """Velocity-weighted moving average. v_thr is in signal units per second."""
    def __init__(self, fs=30.0, window_s=0.4, v_thr=None, floor=0.1):
        self.fs, self.n, self.v_thr, self.floor = fs, max(2, int(window_s * fs)), v_thr, floor
        self.buf, self.wbuf, self.prev = deque(maxlen=self.n), deque(maxlen=self.n), None
        self.vabs = deque(maxlen=int(3 * fs))

    def __call__(self, v):
        vel = 0.0 if self.prev is None else abs(v - self.prev) * self.fs
        self.prev = v
        self.vabs.append(vel)
        thr = self.v_thr or (2.0 * float(np.median(self.vabs)) + 1e-9)
        w = 1.0 if vel <= thr else self.floor
        self.buf.append(v); self.wbuf.append(w)
        b, ww = np.array(self.buf), np.array(self.wbuf)
        return float((b * ww).sum() / ww.sum())


class AdaptChain:
    """Nullpoint's live chain for one scalar signal.

    With a profile frequency f: [optional FLC cancel] -> one velocity-weighted (foveation) average
    over exactly k periods (period-locked window). A single window instead of two in series:
    the series version lagged ~2 periods, which at 2 Hz made the auto-repeat overshoot into the
    next hold (found in the replay benchmark). Without a profile: a short plain moving average."""

    def __init__(self, fs=30.0, freq=None, use_wflc=False, use_fov=True, k_periods=1):
        self.freq = freq
        self.hp = HighPass(fs, 0.8)
        self.wflc = FLC(fs, f0=freq) if (use_wflc and freq) else None
        win = k_periods / freq if freq else 0.33
        self.avg = FoveationMean(fs, window_s=win) if use_fov else PeriodMean(fs, freq, k=k_periods)

    def __call__(self, s):
        if self.wflc is not None:
            s = s - self.wflc(self.hp(s))
        return self.avg(s)


def apply(filt, x):
    return np.array([filt(v) for v in x])
