"""Using Nullpoint's adaptation layer in your own gaze app: a self-contained example.

    python examples/stabilize_any_gaze.py

A made-up gaze stream (someone looking at three targets, 15 deg apart, 4 s each, with 10 deg of
jerk nystagmus at 3 Hz) goes through three pipelines: raw, ordinary smoothing (a 0.1 s moving
average, roughly what gaze apps do), and GazeStabilizer with the person's profile. It counts how
often each pipeline's gaze crosses into a neighbouring target's zone: every crossing is a
potential wrong selection in any gaze app.
"""
import numpy as np

from nullpoint.adapt import GazeStabilizer
from nullpoint.adapt.filters import PeriodMean
from nullpoint.scope import synth

FS, HOLD_S = 30.0, 4.0
TARGETS = [-15.0, 0.0, 15.0, 0.0, -15.0, 15.0]          # degrees

_, nyst, _ = synth.generate(synth.NystParams(kind="jerk_r", freq_hz=3.0, amp_deg=10.0), len(TARGETS) * HOLD_S, FS, seed=3)
intended = np.repeat(TARGETS, int(HOLD_S * FS))
raw = intended + nyst[: len(intended)]

profile = {"freq_hz": 3.0, "null_deg": 0.0}          # what Scope measures (or GazeStabilizer.from_profile())
stab = GazeStabilizer.from_profile(profile, fs=FS)
smooth = PeriodMean(FS, None, default_s=0.1)
out = {"raw": raw, "ordinary smoothing": np.array([smooth(v) for v in raw]),
       "Nullpoint": np.array([stab(v) for v in raw])}


def zone(d):                                          # targets at -15 / 0 / +15: boundaries at -7.5 / +7.5
    return np.digitize(d, [-7.5, 7.5])


settled = np.concatenate([np.arange(int(i * HOLD_S * FS) + 15, int((i + 1) * HOLD_S * FS)) for i in range(len(TARGETS))])
want = zone(intended)
print(f"{'pipeline':20s} {'frames in the wrong zone':>26s} {'zone changes while holding':>28s}")
for name, g in out.items():
    z = zone(g)
    wrong = np.mean(z[settled] != want[settled])
    flips = int(np.sum(np.diff(z[settled]) != 0) - (len(TARGETS) - 1))
    print(f"{name:20s} {wrong:>25.0%} {max(flips, 0):>28d}")
print(f"\nNullpoint dwell time: {stab.dwell_s:.2f} s (one 3 Hz period); put key targets at screen x = {stab.null_screen_x:.2f}")
