"""Load calibration sessions into feature arrays."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nullpoint.capture import FaceFrame
from nullpoint.features import gaze_features, blink_features, N_GAZE, N_BLINK


@dataclass
class Session:
    name: str
    t: np.ndarray
    G: np.ndarray          # (N, N_GAZE)  NaN rows where no face
    B: np.ndarray          # (N, N_BLINK)
    block: np.ndarray
    target: np.ndarray
    cue: np.ndarray
    fs: float


def load_session(path: Path) -> Session:
    d = dict(np.load(path, allow_pickle=False))   # materialise once (NpzFile indexing re-decompresses)
    keys = [str(k) for k in d["blend_keys"]]
    wh = tuple(int(v) for v in d["wh"])
    G = np.full((len(d["t"]), N_GAZE), np.nan, np.float32)
    B = np.full((len(d["t"]), N_BLINK), np.nan, np.float32)
    for i in range(len(d["t"])):
        if not d["has_face"][i]:
            continue
        bl = dict(zip(keys, d["blend"][i])) if keys else {}
        pose = d["pose"][i] if np.isfinite(d["pose"][i]).all() else None
        ff = FaceFrame(float(d["t"][i]), d["lms"][i].astype(np.float32), bl, pose, wh, float(d["brightness"][i]))
        G[i] = gaze_features(ff); B[i] = blink_features(ff)
    t = d["t"]
    return Session(Path(path).stem, t, G, B, d["block"].astype(str), d["target"].astype(str), d["cue"],
                   fs=float(1 / np.median(np.diff(t))))
