"""Nullpoint's adaptation layer as a drop-in for any gaze app.

    from nullpoint.adapt import GazeStabilizer
    stab = GazeStabilizer.from_profile()          # data/profiles/profile.json, or defaults
    x_steady = stab(x_raw)                        # call once per gaze sample, in any unit
    stab.dwell_s, stab.null_screen_x              # dwell time and where to put key targets

Whatever produces the gaze samples (a webcam model, a commercial eye tracker, a headset), the
layer does the same three things the board does:
  * filter: a velocity-weighted (foveation) average over exactly one nystagmus period, so each
    beat of the oscillation cancels and the moments the eyes are briefly still count most
  * dwell: a whole number of nystagmus periods (>= 0.3 s), so a selection never ends mid-beat
  * layout: the screen position of the null point, where the eyes are calmest
Without a measured frequency it falls back to a 0.33 s foveation-weighted window.
"""
from __future__ import annotations

import math

from nullpoint.adapt.filters import AdaptChain
from nullpoint.config import load_profile, null_screen_x


class GazeStabilizer:
    def __init__(self, freq_hz: float | None = None, null_deg: float | None = None, fs: float = 30.0):
        self.freq_hz, self.null_deg, self.fs = freq_hz, null_deg, fs
        self._x = AdaptChain(fs, freq_hz, use_wflc=False)
        self._y = AdaptChain(fs, freq_hz, use_wflc=False)

    @classmethod
    def from_profile(cls, profile: dict | str | None = None, fs: float = 30.0) -> "GazeStabilizer":
        """A profile dict, a path to one, or None for data/profiles/profile.json (if present)."""
        p = profile if isinstance(profile, dict) else load_profile(profile)
        freq = p.get("freq_hz") if p.get("n_periodic_holds", 1) else None
        return cls(freq_hz=freq, null_deg=p.get("null_deg"), fs=fs)

    def __call__(self, x: float, y: float | None = None):
        """One gaze sample in, one steadied sample out (x alone, or (x, y))."""
        sx = self._x(float(x))
        return sx if y is None else (sx, self._y(float(y)))

    @property
    def dwell_s(self) -> float:
        f = self.freq_hz
        return math.ceil(0.3 * f) / f if f else 0.4

    @property
    def null_screen_x(self) -> float:
        return null_screen_x({"null_deg": self.null_deg} if self.null_deg is not None else {})
