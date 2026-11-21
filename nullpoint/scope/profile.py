"""Build a personal nystagmus profile from tracked slow-mo clips.

    python -m nullpoint.scope.profile data/raw/slowmo/a.track.npz [b.track.npz ...] \
        [--targets data/raw/targets-*.json] [--out data/profiles/me.json]

Steps per clip: split the eye trace into steady holds (cut at large saccades / tracking
re-cuts), analyse each hold >= 1.5 s with `metrics.analyse`, and (if a targets log is given
and the number of holds matches) label holds with the commanded gaze angle.
Across holds: robust summary (median frequency/amplitude, majority direction/waveform) and,
when holds at >= 3 gaze angles exist, the null point by Gaussian-process regression of
intensity vs gaze angle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nullpoint.config import PROFILES
from nullpoint.scope.slowmo import Track
from nullpoint.scope import metrics


def split_holds(x: np.ndarray, fs: float, min_len_s: float = 1.5, step_deg: float = 4.0):
    """Cut where the 150 ms-smoothed position jumps by > step_deg within 150 ms (gaze shifts)."""
    k = max(3, int(0.15 * fs))
    sm = np.convolve(np.nan_to_num(x, nan=np.nanmedian(x)), np.ones(k) / k, mode="same")
    jump = np.abs(sm[k:] - sm[:-k]) > step_deg
    cuts = np.flatnonzero(np.diff(jump.astype(int)) == 1) + k
    edges = np.r_[0, cuts, len(x)]
    holds = []
    for a, b in zip(edges[:-1], edges[1:]):
        a2, b2 = a + int(0.25 * fs), b - int(0.1 * fs)      # drop the settling part
        if b2 - a2 >= min_len_s * fs:
            holds.append(slice(a2, b2))
    return holds


def null_point(angles, intensity):
    """GP regression of intensity vs gaze angle -> (null_deg, ci_lo, ci_hi, grid, mean, std)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    a = np.asarray(angles, float)[:, None]; y = np.asarray(intensity, float)
    ym = y.mean()
    gp = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(10.0, (3.0, 60.0)) + WhiteKernel(0.1),
                                  normalize_y=True, random_state=0).fit(a, y - ym)
    grid = np.linspace(a.min(), a.max(), 121)[:, None]
    mu, sd = gp.predict(grid, return_std=True); mu += ym
    # uncertainty of the minimum location: sample posterior functions
    samp = gp.sample_y(grid, 400, random_state=0) + ym
    mins = grid[np.argmin(samp, axis=0), 0]
    return float(grid[np.argmin(mu), 0]), float(np.percentile(mins, 10)), float(np.percentile(mins, 90)), grid[:, 0], mu, sd


def build(tracks: list[Path], targets: list[Path] | None = None) -> dict:
    holds_out = []
    for i, p in enumerate(tracks):
        tr = Track.load(p)
        e = tr.eye_deg(head_comp=True)[:, 0]
        seq = None
        if targets and i < len(targets):
            seq = [s["deg"] for s in json.loads(Path(targets[i]).read_text())["sequence"]]
        holds = []
        for sg in tr.segments:
            for h in split_holds(e[sg], tr.fps):
                holds.append(slice(sg.start + h.start, sg.start + h.stop))
        labelled = seq is not None and len(holds) == len(seq)
        for j, h in enumerate(holds):
            m = metrics.analyse(e[h], tr.fps).dict()
            try:
                from nullpoint.scope.waveform_cnn import classify
                lab, pr = classify(e[h], tr.fps, fs_model=120)
                m["waveform_cnn"], m["waveform_cnn_p"] = lab, [round(float(v), 3) for v in pr]
            except Exception as ex:          # weights missing etc.: keep the heuristic only
                m["waveform_cnn"], m["waveform_cnn_p"] = None, str(ex)
            m.update(clip=p.stem, start_s=round(h.start / tr.fps, 2), dur_s=round((h.stop - h.start) / tr.fps, 2),
                     gaze_deg=seq[j] if labelled else None)
            holds_out.append(m)
    good = [h for h in holds_out if h["periodicity"] > 0.25]
    def med(k, src):
        v = [h[k] for h in src if h[k] is not None and np.isfinite(h[k])]
        return float(np.median(v)) if v else None
    from collections import Counter
    prof = {
        "n_holds": len(holds_out), "n_periodic_holds": len(good),
        "freq_hz": med("freq_hz", good), "amp_deg": med("amp_deg", good), "intensity_deg_s": med("intensity", good),
        "fast_dir": Counter(h["fast_dir"] for h in good).most_common(1)[0][0] if good else None,
        "waveform": Counter(h["waveform_heuristic"] for h in good).most_common(1)[0][0] if good else None,
        "waveform_cnn": Counter(h.get("waveform_cnn") for h in good).most_common(1)[0][0] if good else None,
        "foveation_frac": med("foveation_frac", good),
        "holds": holds_out,
    }
    lab = [h for h in good if h["gaze_deg"] is not None]
    angles = sorted({h["gaze_deg"] for h in lab})
    if len(angles) >= 3:
        nd, lo, hi, *_ = null_point([h["gaze_deg"] for h in lab], [h["intensity"] for h in lab])
        prof["null_deg"], prof["null_ci_deg"] = nd, [lo, hi]
    else:
        prof["null_deg"], prof["null_ci_deg"] = None, None
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", type=Path, nargs="+")
    ap.add_argument("--targets", type=Path, nargs="*")
    ap.add_argument("--out", type=Path, default=PROFILES / "profile.json")
    a = ap.parse_args()
    prof = build(a.tracks, a.targets)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(prof, indent=1))
    show = {k: v for k, v in prof.items() if k != "holds"}
    print(json.dumps(show, indent=1))
    for h in prof["holds"]:
        print(f"  {h['clip']:14s} t={h['start_s']:6.2f}s dur={h['dur_s']:5.2f}  f={h['freq_hz']:.2f}Hz (welch {h['freq_welch_hz']:.2f})"
              f"  per={h['periodicity']:.2f}  amp={h['amp_deg']:.2f}deg  dir={h['fast_dir']:5s} wf={h['waveform_heuristic']} cnn={h.get('waveform_cnn')}")


if __name__ == "__main__":
    main()
