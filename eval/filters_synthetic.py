"""Compare gaze-steadying filters on synthetic nystagmus at 30 fps (webcam rate).

Signal = intended gaze (0 deg, stepping to 10 deg at t = 5 s) + synthetic nystagmus + 0.3 deg noise.
Metric = RMSE of the filtered signal against the intended gaze, excluding 1 s after the step.
Averaged over waveforms (pendular, jerk_r), frequencies (2, 3, 4.5 Hz), amplitudes (3, 6, 10 deg)
and 3 seeds. Writes reports/filters_synthetic.json.
"""
import itertools
import json

import numpy as np

from nullpoint.config import REPORTS
from nullpoint.scope import synth
from nullpoint.adapt.filters import OneEuro, PeriodMean, FoveationMean, AdaptChain, apply

FS = 30.0


class _Series:
    def __init__(self, *fs): self.fs = fs
    def __call__(self, v):
        for f in self.fs: v = f(v)
        return v


def filters(f):
    return {
        "raw": lambda: (lambda v: v),
        "one-euro (baseline)": lambda: OneEuro(FS),
        "moving mean 0.33 s (baseline)": lambda: PeriodMean(FS, None),
        "period-locked mean": lambda: PeriodMean(FS, f),
        "foveation-weighted mean": lambda: FoveationMean(FS, 1 / f),
        "foveation-weighted, one period (Nullpoint)": lambda: AdaptChain(FS, f),
        "foveation then period mean (series, v1)": lambda: _Series(FoveationMean(FS, 1 / f), PeriodMean(FS, f)),
        "FLC + Nullpoint": lambda: AdaptChain(FS, f, use_wflc=True),
    }


def main():
    rows = {}
    for kind, f, amp, seed in itertools.product(("pendular", "jerk_r"), (2.0, 3.0, 4.5), (3.0, 6.0, 10.0), range(3)):
        t, x, _ = synth.generate(synth.NystParams(kind=kind, freq_hz=f, amp_deg=amp), 12, FS, seed=seed, noise_deg=0.3)
        target = np.where(t < 5, 0.0, 10.0); s = target + x
        m = (t > 2) & ((t < 5) | (t > 6))
        for name, mk in filters(f).items():
            y = apply(mk(), s)
            rows.setdefault(name, {}).setdefault(kind, []).append(float(np.sqrt(np.mean((y[m] - target[m]) ** 2))))
            # lag: time after the 10 deg step until the output first crosses 5 deg
            k = np.flatnonzero((t >= 5) & (y > 5.0))
            rows[name].setdefault("lag", []).append(float(t[k[0]] - 5) if len(k) else np.nan)
    out = {n: {k: round(float(np.nanmean(v)), 3) for k, v in d.items()}
              | {"all": round(float(np.mean(d["pendular"] + d["jerk_r"])), 3)} for n, d in rows.items()}
    for n, d in out.items():
        print(f"{n:42s} pendular {d['pendular']:.2f}  jerk {d['jerk_r']:.2f}  all {d['all']:.2f} deg RMSE   lag {d['lag']*1000:4.0f} ms")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "filters_synthetic.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
