"""Starter models so the live board can run before a proper calibration.

    python -m nullpoint.models.quickstart

Glance: LDA trained on the existing grid scans (data/raw/scan-*-grid-*.npz), middle row:
dots at screen x 0.1 / 0.5 / 0.9 -> L / C / R. Those scans have no blendshapes or head pose,
so the model uses only the landmark-geometry eye features (iris position and lid opening of both
eyes; no head position, which shifted between sessions: eval/recal.py).
Blink: no labelled blinks exist yet, so a fixed rule: a blink lasting >= 0.45 s counts as
deliberate. Natural blinks last ~0.1-0.3 s.

These are stand-ins. Run `python -m nullpoint.calibrate` + `python -m nullpoint.models.train`
for real per-user models (that overwrites these).
"""
import json
import sys
from pathlib import Path

import numpy as np

from nullpoint.config import RAW, TRAINED
from nullpoint.models.glance import GlanceModel, FEATURE_SETS
from nullpoint.models.blink import rule_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from replay import load_scan, train_frames, LAYOUTS  # noqa: E402

GEOM_COLS = FEATURE_SETS["eye_geom"]      # r_ix r_iy r_open l_ix l_iy l_open
BLINK_TAU_S = 0.45


def main():
    scans = sorted(RAW.glob("scan-*-grid-*.npz"))
    if not scans:
        raise SystemExit("no grid scans in data/raw - run python -m nullpoint.calibrate instead")
    G, y = [], []
    for p in scans:
        g, yy = train_frames(*load_scan(p), zx=LAYOUTS["wide"])
        G.append(g); y.append(yy)
    G, y = np.vstack(G), np.concatenate(y)
    gm = GlanceModel("lda", cols=GEOM_COLS).fit(G, y)
    gm.zone_x = {"L": 0.1, "C": 0.5, "R": 0.9}         # the grid-scan dots it was trained on
    bm = rule_model(BLINK_TAU_S)
    TRAINED.mkdir(parents=True, exist_ok=True)
    gm.save(TRAINED / "glance.pkl"); bm.save(TRAINED / "blink.pkl")
    info = {"quickstart": True, "glance_from": [p.name for p in scans], "glance_frames": int(len(y)),
            "glance_features": "eye geometry (iris position, lid opening)", "blink": f"duration >= {BLINK_TAU_S} s"}
    (TRAINED / "info.json").write_text(json.dumps(info, indent=1))
    acc = float((gm.proba(G).argmax(1) == y).mean())
    print(json.dumps(info | {"train_acc": round(acc, 3)}, indent=1))


if __name__ == "__main__":
    main()
