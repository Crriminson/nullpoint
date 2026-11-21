"""Fit the user's glance + blink models from calibration sessions and save them.

    python -m nullpoint.models.train                   # all data/calib/*.npz
    python -m nullpoint.models.train --glance lgbm --blink gru
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nullpoint.config import CALIB, TRAINED
from nullpoint.models.data import load_session
from nullpoint.models.glance import GlanceModel, glance_frames, FEATURE_SETS
from nullpoint.config import zone_targets, null_screen_x, load_profile
from nullpoint.models.blink import detect_events, label_events, BlinkModel, DurationThreshold, make_lgbm, GRUBlink


def blink_dataset(sessions):
    X, S, y = [], [], []
    for s in sessions:
        ev = detect_events(s.t, s.B)
        lab = label_events(s, ev)
        for e, l in zip(ev, lab):
            if l >= 0:
                X.append(e.feats); S.append(e.seq); y.append(l)
    return np.array(X), np.array(S), np.array(y)


def make_blink(kind):
    return {"threshold": DurationThreshold(), "lgbm": make_lgbm(), "gru": GRUBlink()}[kind]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glance", default="lda", choices=["lda", "lgbm", "threshold"])
    ap.add_argument("--features", default="eye", choices=list(FEATURE_SETS),
                    help="eye = eye-in-head features only (default; head position drifts between sessions)")
    ap.add_argument("--blink", default="lgbm", choices=["threshold", "lgbm", "gru"])
    ap.add_argument("--files", type=Path, nargs="*")
    a = ap.parse_args()
    files = a.files or sorted(CALIB.glob("calib-*.npz"))
    if not files:
        raise SystemExit("no calibration sessions in data/calib - run: python -m nullpoint.calibrate")
    sess = [load_session(f) for f in files]
    G = np.vstack([glance_frames(s)[0] for s in sess]); yg = np.concatenate([glance_frames(s)[1] for s in sess])
    gm = GlanceModel(a.glance, cols=FEATURE_SETS[a.features]).fit(G, yg)
    gm.zone_x = zone_targets(null_screen_x(load_profile()))    # where calibrate.py drew the dots
    X, S, yb = blink_dataset(sess)
    clf = make_blink(a.blink)
    clf.fit(S if a.blink == "gru" else X, yb)
    bm = BlinkModel(clf, a.blink)
    TRAINED.mkdir(parents=True, exist_ok=True)
    gm.save(TRAINED / "glance.pkl"); bm.save(TRAINED / "blink.pkl")
    info = {"sessions": [f.name for f in files], "glance_kind": a.glance, "glance_features": a.features, "glance_frames": int(len(yg)),
            "blink_kind": a.blink, "blink_events": int(len(yb)), "deliberate": int(yb.sum())}
    (TRAINED / "info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
