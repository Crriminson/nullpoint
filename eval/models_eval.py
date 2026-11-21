"""Cross-session evaluation of the glance and blink models (train on one session, test on another).

    python eval/models_eval.py                      # all ordered pairs of data/calib sessions

Glance: per-frame accuracy and HMM-filtered accuracy, threshold vs LDA vs LightGBM.
Blink : deliberate-vs-natural F1 and false-trigger rate (natural blinks classified deliberate,
        per minute of the natural/reading block), duration threshold vs LightGBM vs GRU.
Writes reports/models_eval.json.
"""
from __future__ import annotations

import itertools
import json

import numpy as np
from sklearn.metrics import f1_score

from nullpoint.config import CALIB, REPORTS
from nullpoint.models.data import load_session
from nullpoint.models.glance import GlanceModel, glance_frames
from nullpoint.models.blink import detect_events, label_events
from nullpoint.models.train import make_blink


def blink_set(s):
    ev = detect_events(s.t, s.B); lab = label_events(s, ev)
    keep = lab >= 0
    X = np.array([e.feats for e in ev])[keep]; S = np.array([e.seq for e in ev])[keep]
    blocks = np.array([s.block[e.i0] for e in ev])[keep]
    return X, S, lab[keep], blocks


def main():
    files = sorted(CALIB.glob("calib-*.npz"))
    if len(files) < 2:
        raise SystemExit("need >= 2 calibration sessions for cross-session evaluation")
    sess = {f.stem: load_session(f) for f in files}
    out = {"glance": [], "blink": []}
    for a, b in itertools.permutations(sess, 2):
        A, B = sess[a], sess[b]
        Ga, ya, _ = glance_frames(A); Gb, yb, _ = glance_frames(B)
        for kind in ("threshold", "lda", "lgbm"):
            m = GlanceModel(kind).fit(Ga, ya)
            P = m.proba(Gb)
            acc = float((P.argmax(1) == yb).mean())
            acc_hmm = float((m.hmm.filter(P).argmax(1) == yb).mean())
            out["glance"].append({"train": a, "test": b, "model": kind, "frame_acc": acc, "hmm_acc": acc_hmm})
            print(f"glance {kind:9s} {a[-12:]} -> {b[-12:]}: frame {acc:.3f}  +HMM {acc_hmm:.3f}")
        Xa, Sa, la, _ = blink_set(A); Xb, Sb, lb, blk = blink_set(B)
        if len(set(la)) < 2 or len(set(lb)) < 2:
            continue
        nat_minutes = (B.block == "natural").sum() / B.fs / 60
        for kind in ("threshold", "lgbm", "gru"):
            clf = make_blink(kind).fit(Sa if kind == "gru" else Xa, la)
            p = clf.predict_proba(Sb if kind == "gru" else Xb)[:, 1]
            pred = (p >= 0.5).astype(int)
            fp_nat = int(((pred == 1) & (lb == 0) & (blk == "natural")).sum())
            r = {"train": a, "test": b, "model": kind, "f1": float(f1_score(lb, pred)),
                 "recall_deliberate": float(pred[lb == 1].mean()), "n_test": int(len(lb)),
                 "false_triggers_per_min_reading": fp_nat / nat_minutes if nat_minutes else None}
            out["blink"].append(r)
            print(f"blink  {kind:9s} {a[-12:]} -> {b[-12:]}: F1 {r['f1']:.3f} recall {r['recall_deliberate']:.2f} "
                  f"false/min(reading) {r['false_triggers_per_min_reading']:.2f}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "models_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
