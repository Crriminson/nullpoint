"""Validate Scope metrics on synthetic nystagmus with known ground truth.

For random clinically-plausible parameters (freq 1.2-6 Hz, amp 0.5-15 deg p-p, all waveform
families) at a given frame rate and measurement noise, compare the measured frequency,
amplitude, fast-phase direction and waveform (heuristic) with the truth.
Writes reports/scope_synthetic.json and a figure.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nullpoint.config import REPORTS
from nullpoint.scope import synth, metrics


def run(fs: float, noise: float, n: int = 300, seed: int = 0, dur: float = 6.0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        kind = synth.CLASSES[1 + i % 3]   # only oscillating classes for freq/amp error
        p = synth.random_params(rng, kind)
        _, x, fov = synth.generate(p, dur, fs, seed=int(rng.integers(1 << 31)), noise_deg=noise, drift_deg=0.5)
        m = metrics.analyse(x, fs)
        true_dir = {"jerk_r": "right", "jerk_l": "left"}.get(kind, "none")
        rows.append(dict(kind=kind, f_true=p.freq_hz, a_true=p.amp_deg, f=m.freq_hz, fw=m.freq_welch_hz, a=m.amp_deg,
                         dir_ok=(m.fast_dir == true_dir) if kind != "pendular" else None, wf=m.waveform_heuristic,
                         gaze_mean=m.gaze_mean_deg, gaze_fov=m.gaze_foveation_deg))
    return rows


def summarise(rows):
    f_t = np.array([r["f_true"] for r in rows]); f = np.array([r["f"] for r in rows]); fw = np.array([r["fw"] for r in rows])
    a_t = np.array([r["a_true"] for r in rows]); a = np.array([r["a"] for r in rows])
    ok = np.isfinite(f)
    rel_f = np.abs(f[ok] - f_t[ok]) / f_t[ok]
    rel_fw = np.abs(fw - f_t) / f_t
    rel_a = np.abs(a - a_t) / a_t
    jerk = [r for r in rows if r["kind"] != "pendular"]
    # gaze estimate error: the true target is 0 deg in synth coordinates
    gm = np.array([abs(r["gaze_mean"]) for r in jerk]); gf = np.array([abs(r["gaze_fov"]) for r in jerk])
    big = a_t > 1.0
    return {
        "n": len(rows),
        "freq_autocorr_within_10pct": float(np.mean(rel_f < 0.10)) * ok.mean(),
        "freq_autocorr_median_abs_err_hz": float(np.median(np.abs(f[ok] - f_t[ok]))),
        "freq_welch_within_10pct": float(np.mean(rel_fw < 0.10)),
        "amp_median_rel_err": float(np.median(rel_a)),
        "amp_median_rel_err_amp_gt_1deg": float(np.median(rel_a[big])),
        "fast_dir_acc_jerk": float(np.mean([r["dir_ok"] for r in jerk])),
        "waveform_heuristic_acc": float(np.mean([r["wf"] == r["kind"] for r in rows])),
        "jerk_gaze_err_mean_deg": float(np.median(gm)),
        "jerk_gaze_err_foveation_deg": float(np.median(gf)),
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    out = {}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for fs, noise, c in ((240, 0.2, "C0"), (120, 0.3, "C1"), (30, 0.8, "C2")):
        rows = run(fs, noise)
        s = summarise(rows); out[f"{fs}fps_noise{noise}"] = s
        print(f"{fs:>3} fps, noise {noise} deg:", json.dumps({k: round(v, 3) for k, v in s.items()}))
        ax[0].scatter([r["f_true"] for r in rows], [r["f"] for r in rows], s=6, c=c, label=f"{fs} fps")
        ax[1].scatter([r["a_true"] for r in rows], [r["a"] for r in rows], s=6, c=c, label=f"{fs} fps")
    ax[0].plot([1, 6], [1, 6], "k--", lw=.8); ax[0].set_xlabel("true frequency (Hz)"); ax[0].set_ylabel("measured"); ax[0].legend()
    ax[1].plot([0.5, 15], [0.5, 15], "k--", lw=.8); ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("true amplitude (deg p-p)"); ax[1].set_ylabel("measured")
    # example traces
    for k, kind in enumerate(["pendular", "jerk_r"]):
        t, x, fov = synth.generate(synth.NystParams(kind=kind, freq_hz=3, amp_deg=4), 1.5, 240, seed=3)
        ax[2].plot(t, x + 6 * k, lw=1, label=kind); ax[2].plot(t[fov], x[fov] + 6 * k, ".", ms=2, c="k")
    ax[2].set_title("synthetic examples (black = foveation)"); ax[2].legend()
    plt.tight_layout(); plt.savefig(REPORTS / "scope_synthetic.png", dpi=90)
    (REPORTS / "scope_synthetic.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
