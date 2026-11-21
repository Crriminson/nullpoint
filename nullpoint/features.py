"""Per-frame features for gaze and blink models.

Gaze features (scale-free): for each eye, iris centre projected on the eye-corner axis and its
normal (in eye-widths), lid opening; head yaw/pitch/roll from the facial transformation matrix;
nose offset from the eye midpoint (head-pose proxy that also works without the matrix).
Blink features: MediaPipe blink/squint blendshapes + lid opening.
"""
from __future__ import annotations

import numpy as np

from nullpoint.config import R_OUT, R_IN, R_IRIS, R_TOP, R_BOT, L_OUT, L_IN, L_IRIS, L_TOP, L_BOT, NOSE_TIP

GAZE_NAMES = ["r_ix", "r_iy", "r_open", "l_ix", "l_iy", "l_open", "yaw", "pitch", "roll", "nose_x", "nose_y",
              "lookInL", "lookOutL", "lookInR", "lookOutR", "lookUpL", "lookDownL"]
BLINK_NAMES = ["blinkL", "blinkR", "squintL", "squintR", "r_open", "l_open", "browDownL", "browDownR"]
N_GAZE, N_BLINK = len(GAZE_NAMES), len(BLINK_NAMES)


def _eye(L, out, inn, iris, top, bot, wh):
    P = L[:, :2] * np.asarray(wh, np.float32)
    a, b = P[out], P[inn]
    w = float(np.linalg.norm(b - a)) + 1e-6
    u = (b - a) / w; n = np.array([-u[1], u[0]])
    rel = P[iris] - a
    # x measured from the outer corner towards the inner one -> mirror the left eye so both
    # eyes increase in the same screen direction
    return float(rel @ u / w), float(rel @ n / w), float(np.linalg.norm(P[top] - P[bot]) / w)


def euler_from_matrix(M: np.ndarray) -> tuple[float, float, float]:
    R = np.asarray(M)[:3, :3]
    sy = np.hypot(R[0, 0], R[1, 0])
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return float(yaw), float(pitch), float(roll)


def gaze_features(ff) -> np.ndarray | None:
    if ff.lms is None:
        return None
    L, wh = ff.lms, ff.wh
    rx, ry, ro = _eye(L, R_OUT, R_IN, R_IRIS, R_TOP, R_BOT, wh)
    lx, ly, lo = _eye(L, L_OUT, L_IN, L_IRIS, L_TOP, L_BOT, wh)
    lx = 1.0 - lx    # mirror: both eyes now increase towards image-right
    rx = rx
    yaw, pitch, roll = euler_from_matrix(ff.pose) if ff.pose is not None else (0.0, 0.0, 0.0)
    P = L[:, :2] * np.asarray(wh, np.float32)
    mid = P[[R_OUT, R_IN, L_OUT, L_IN]].mean(0); iod = float(np.linalg.norm(P[R_OUT] - P[L_OUT])) + 1e-6
    nx, ny = (P[NOSE_TIP] - mid) / iod
    b = ff.blend or {}
    return np.array([rx, ry, ro, lx, ly, lo, yaw, pitch, roll, nx, ny,
                     b.get("eyeLookInLeft", 0), b.get("eyeLookOutLeft", 0), b.get("eyeLookInRight", 0),
                     b.get("eyeLookOutRight", 0), b.get("eyeLookUpLeft", 0), b.get("eyeLookDownLeft", 0)], np.float32)


def blink_features(ff) -> np.ndarray | None:
    if ff.lms is None:
        return None
    b = ff.blend or {}
    _, _, ro = _eye(ff.lms, R_OUT, R_IN, R_IRIS, R_TOP, R_BOT, ff.wh)
    _, _, lo = _eye(ff.lms, L_OUT, L_IN, L_IRIS, L_TOP, L_BOT, ff.wh)
    return np.array([b.get("eyeBlinkLeft", 0), b.get("eyeBlinkRight", 0), b.get("eyeSquintLeft", 0),
                     b.get("eyeSquintRight", 0), ro, lo, b.get("browDownLeft", 0), b.get("browDownRight", 0)], np.float32)


def horizontal_gaze(g: np.ndarray) -> float:
    """1-D gaze signal used by the oscillation filters: mean of both eyes' iris x."""
    return float((g[0] + g[3]) / 2)
