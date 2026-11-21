"""Cross-session drift: which glance features survive a new session, and does a 10-second setup or
self-correction fix what doesn't?

The replay benchmark (eval/replay.py --cross) found that, in the narrow layout, a glance model
trained on one grid scan and used on the next made ~16 wrong steps per minute with no nystagmus at
all. Two fixes are compared, on the same scans, selector, scoring, nystagmus injection and seeds as
the replay benchmark (Adapt on, as the board runs):

Glance model features
  all : every landmark feature, including head position in the image (nose)
  eye : eye-in-head only (iris position and lid opening of both eyes). The grid scans have no
        blendshapes, so this is FEATURE_SETS["eye_geom"]; the board's default "eye" set adds the
        gaze blendshapes.

Adaptation arms (GlanceModel alignment; nothing is retrained)
  none        : the scan-A model as is
  setup       : offset alignment from B's dots at the zone positions, always applied.
                LEAVE-ONE-HOLD-OUT: the hold being scored is never in the setup data (conservative:
                the real setup also shows that position)
  setup-gated : the same, with the do-no-harm rule the board uses: keep the trained state if it
                already classifies >= 95 % of the setup frames correctly
  self-correct: nudges from centre-hold frames only (as at selections), repeated 10 times with the
                board's rules (skip if moving / nothing to correct). The centre hold is nudged with its
                frames 0.8-2.0 s and scored from 2.0 s on; the L / R holds are scored as usual.

New-user arm
  10s-model   : no trained model at all. A fresh model is fitted from setup-sized data of scan A
                (frames 0.8-2.5 s of each L / C / R hold: what the board's 10-second setup collects)
                and used on scan B, the next session. Compare with "none", trained on the full holds.

Also prints a per-frame check (no selector, no nystagmus) of offset vs offset+gain alignment,
leaving one zone out. Writes reports/recal.json.
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from nullpoint.config import RAW, REPORTS
from nullpoint.models.glance import GlanceModel, FEATURE_SETS

sys.path.insert(0, str(Path(__file__).parent))
from replay import load_scan, train_frames, gains, run_raw, summarize, LAYOUTS, REACTION_S  # noqa: E402

SETTLE = 0.8
SPLIT = 2.0                      # self-correct arm: centre-hold frames before this nudge, after it score
AMPS = [0, 8, 15]
KINDS = ["jerk_r", "pendular"]
FEATS = {"all": FEATURE_SETS["all"], "eye": FEATURE_SETS["eye_geom"]}
RUNS = [(f, s) for s in range(2) for f in (2.0, 3.0, 4.0)]
ARMS = ["none", "setup", "setup-gated", "self-correct", "10s-model"]
SETUP_WINDOW = (0.8, 2.5)        # per dot in the board's setup: 2.5 s, first 0.8 s ignored

_W = {}


def hold_frames(scan, x, lo=SETTLE, hi=None):
    t, tx, row, G = scan
    idx = np.flatnonzero(row & (tx == x))
    rel = t[idx] - t[idx[0]]
    keep = idx[(rel >= lo) & ((rel < hi) if hi else True)]
    return G[keep][np.isfinite(G[keep]).all(1)]


def recal_data(scan, zx, xs, hi=None):
    Gs, ys = [], []
    for x in xs:
        g = hold_frames(scan, x, hi=hi)
        Gs.append(g); ys.append(np.full(len(g), zx[x]))
    return np.vstack(Gs), np.concatenate(ys)


def _init():
    scans = sorted(RAW.glob("scan-*-grid-*.npz"))
    _W["A"], _W["B"] = (load_scan(p) for p in scans[:2])
    _W["gain"] = gains(*_W["A"])


def _job(args):
    lay, feat, kind, amp = args
    A, B, gain = _W["A"], _W["B"], _W["gain"]
    zx = LAYOUTS[lay]; xs = list(zx); cx = [x for x in xs if zx[x] == 1][0]
    gm = GlanceModel("lda", cols=FEATS[feat]).fit(*train_frames(*A, zx=zx))
    common = dict(kind=kind, amp=amp, adapt=True, zx=zx)

    def runs(score=None):
        return [run_raw(*B, gm, gain, freq=f, seed=s, score=score, **common) for f, s in RUNS]

    row, info = {}, {}
    gm.reset_alignment(); row["none"] = summarize(runs())
    for arm, keep_if in (("setup", None), ("setup-gated", 0.95)):
        raws, changed = [], 0
        for h in xs:                                   # leave the scored hold out of the setup data
            r = gm.recalibrate(*recal_data(B, zx, [x for x in xs if x != h]), keep_if=keep_if)
            changed += bool(r.get("changed"))
            raws += runs({h: REACTION_S})
        row[arm] = summarize(raws); info[arm + "_changed"] = f"{changed}/{len(xs)}"
    gm.reset_alignment()
    Gc = hold_frames(B, cx, hi=SPLIT)
    n = sum(gm.nudge(Gc[i:i + 20], 1) is not None for i in [0, 10, 20] * 4 if len(Gc[i:i + 20]) >= 5)
    info["self-correct_nudges"] = int(n)
    row["self-correct"] = summarize(runs({x: (SPLIT if x == cx else REACTION_S) for x in xs}))
    Gs, ys = [], []
    for x in xs:                                       # what a 10-second setup would have collected in A
        g = hold_frames(A, x, lo=SETUP_WINDOW[0], hi=SETUP_WINDOW[1]); Gs.append(g); ys.append(np.full(len(g), zx[x]))
    gm = GlanceModel("lda", cols=FEATS[feat]).fit(np.vstack(Gs), np.concatenate(ys))
    info["10s-model_frames"] = int(sum(len(g) for g in Gs))
    row["10s-model"] = summarize(runs())
    return f"{lay}|{feat}|{kind}|{amp}", row, info


def per_frame_check(A, B):
    """Leave-one-zone-out per-frame accuracy (no nystagmus), offset vs offset+gain."""
    out = {}
    for lay, zx in LAYOUTS.items():
        xs = list(zx)
        for feat, cols in FEATS.items():
            gm = GlanceModel("lda", cols=cols).fit(*train_frames(*A, zx=zx))
            for mode in ("none", "offset", "affine"):
                acc = []
                for h in xs:
                    gm.reset_alignment()
                    if mode != "none":
                        gm.recalibrate(*recal_data(B, zx, [x for x in xs if x != h]), mode=mode, keep_if=None)
                    acc.append(float((gm.proba(hold_frames(B, h)).argmax(1) == zx[h]).mean()))
                out[f"{lay}|{feat}|{mode}"] = [round(a, 3) for a in acc]
    return out


def main():
    scans = sorted(RAW.glob("scan-*-grid-*.npz"))
    if len(scans) < 2:
        raise SystemExit("need two grid scans in data/raw")
    _init()
    pf = per_frame_check(_W["A"], _W["B"])
    print("per-frame accuracy per held zone (L, C, R), leave-one-zone-out:")
    for k, v in pf.items():
        print(f"  {k:24s} {v}")
    jobs = [(lay, feat, kind, amp) for lay in LAYOUTS for feat in FEATS for kind in KINDS for amp in AMPS]
    res, infos = {}, {}
    with Pool(4, initializer=_init) as pool:
        for key, row, info in pool.imap(_job, jobs):
            res[key], infos[key] = row, info
            print(f"{key:24s}", "  ".join(f"{a}: {m['hold_acc']:.2f} {m['wrong_per_min']:5.1f}/min {m['latency_s']:.2f}s"
                                          for a, m in row.items()), info, flush=True)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "recal.json").write_text(json.dumps({"train": "scan_A.npz", "test": "scan_B.npz", "adapt": True,
                                                     "freqs_hz": [2, 3, 4], "seeds": 2, "arms": ARMS, "results": res,
                                                     "info": infos, "per_frame_lozo": pf}, indent=1))


if __name__ == "__main__":
    main()
