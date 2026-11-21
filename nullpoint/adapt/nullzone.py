"""Null-zone estimate from a webcam, no slow-motion video needed.

People with infantile nystagmus usually hold their eyes where the oscillation is calmest (the null
zone) when vision gets demanding, turning the head so the eyes can stay there. A 30 fps webcam
cannot see the oscillation (about 0.5 px per degree at 55 cm), but it can see *where in the orbit the eyes
settle* and how the head turns. Three steps, about 40 s:

  1. eyes  : head still and facing the screen, look at three dots at known angles
             -> calibrates eye-in-head angle (mean iris position of both eyes) in degrees
  2. head  : point the nose at two dots -> calibrates head turn (nose offset) in degrees
  3. read  : read shrinking words at the screen centre, head however is easiest

Estimate = median eye-in-head angle while reading (1-s block bootstrap CI). Cross-check: while
fixating the centre, a head turn of +x needs the eyes at -x, so the head estimate is -(head angle).
The two come from different landmarks; if they disagree the result is flagged.

This measures habitual posture, which equals the null zone only if the posture is driven by the
nystagmus. The slow-mo measurement (nullpoint.scope.profile) measures the null directly and is
preferred when available.
"""
from __future__ import annotations

import numpy as np

from nullpoint.config import screen_x_to_deg

STEPS = ([("eyes", "C"), ("eyes", "L"), ("eyes", "R"), ("head", "L"), ("head", "R")]
         + [("read", i) for i in range(12)])
DOT_S, SETTLE_S = 2.5, 0.8
WORD_S, READ_SKIP_S = 2.5, 3.0          # first 3 s of reading: the posture is still settling
WORDS = ["harbour", "violet", "window", "lantern", "meadow", "pocket", "silver", "garden",
         "kettle", "orange", "marble", "candle"]
SIZES = [30, 24, 20, 17, 15, 14, 13, 12, 11, 11, 10, 10]   # CSS px: hard enough to need the null zone
MIN_CLEAR_DEG = 3.0
MIN_HEAD_GAIN = 1e-3                    # nose offset (eye-widths) per degree of head turn


def schedule(el: float):
    """(phase, key, step index, seconds into the step) at `el` seconds, or None when finished."""
    t = 0.0
    for i, (ph, key) in enumerate(STEPS):
        d = WORD_S if ph == "read" else DOT_S
        if el < t + d:
            return ph, key, i, el - t
        t += d
    return None


def total_s() -> float:
    return sum(WORD_S if ph == "read" else DOT_S for ph, _ in STEPS)


def _block_ci(t, x, block_s=1.0, n=500, seed=0):
    """Bootstrap CI of the median, resampling 1-s blocks (frames within a block are correlated)."""
    blocks = [x[(t >= b) & (t < b + block_s)] for b in np.arange(t.min(), t.max() + 1e-9, block_s)]
    blocks = [b for b in blocks if len(b)]
    if len(blocks) < 4:
        return None
    rng = np.random.default_rng(seed)
    meds = [np.median(np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]))
            for _ in range(n)]
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def estimate(rows: list, dot_x: dict) -> dict:
    """rows: (phase, key, seconds_into_step, t, eye, head). dot_x: zone -> screen x of the dots."""
    if not rows:
        return {"ok": False, "reason": "no face seen"}
    ph = np.array([r[0] for r in rows]); key = np.array([str(r[1]) for r in rows])
    into = np.array([r[2] for r in rows], float); t = np.array([r[3] for r in rows], float)
    e = np.array([r[4] for r in rows], float); h = np.array([r[5] for r in rows], float)
    deg = {z: screen_x_to_deg(x) for z, x in dot_x.items()}
    notes = []

    # 1. eye-in-head angle: e = e0 + g * deg, from the three dots with the head still
    pts = []
    for z in ("L", "C", "R"):
        m = (ph == "eyes") & (key == z) & (into >= SETTLE_S)
        if m.sum() >= 10:
            pts.append((deg[z], float(np.median(e[m]))))
    if len(pts) < 3:
        return {"ok": False, "reason": "eyes not seen at every dot"}
    a, ev = np.array(pts).T
    g, e0 = np.polyfit(a, ev, 1)
    if abs(g) < 1e-4 or not (np.all(np.diff(ev) > 0) or np.all(np.diff(ev) < 0)):
        return {"ok": False, "reason": "eye position did not follow the dots"}
    m1 = (ph == "eyes") & (into >= SETTLE_S)
    h0 = float(np.median(h[m1]))
    if np.ptp(np.percentile(h[m1], [10, 90])) > 0.03:
        notes.append("head moved during the eye dots")

    # 2. head turn: h = h0 + gh * deg, from pointing the nose at the side dots
    hm = {z: (ph == "head") & (key == z) & (into >= 1.0) for z in ("L", "R")}
    gh = None
    if all(m.sum() >= 10 for m in hm.values()):
        gh = float((np.median(h[hm["R"]]) - np.median(h[hm["L"]])) / (deg["R"] - deg["L"]))
        if abs(gh) < MIN_HEAD_GAIN:          # a real head turn moves the nose ~0.004 eye-widths per degree
            notes.append("head barely turned when pointing; head check skipped"); gh = None

    # 3. reading: where do the eyes settle?
    read_start = sum(DOT_S for p, _ in STEPS if p != "read")
    mr = (ph == "read") & (t >= read_start + READ_SKIP_S)
    if mr.sum() < 60:
        return {"ok": False, "reason": "too few reading frames (face lost?)"}
    eye_deg = (e[mr] - e0) / g
    null = float(np.median(eye_deg))
    ci = _block_ci(t[mr], eye_deg)
    head_deg = float(np.median((h[mr] - h0) / gh)) if gh else None
    head_check = -head_deg if head_deg is not None else None
    agree = None if head_check is None else bool(abs(head_check - null) <= max(3.0, 0.5 * abs(null)))
    clear = abs(null) >= MIN_CLEAR_DEG and ci is not None and (ci[0] > 0 or ci[1] < 0)
    if agree is False:
        notes.append("eye and head estimates disagree")
    return {"ok": True, "null_deg": round(null, 1), "ci_deg": [round(c, 1) for c in ci] if ci else None,
            "clear": bool(clear), "head_turn_deg": None if head_deg is None else round(head_deg, 1),
            "head_check_deg": None if head_check is None else round(head_check, 1), "agree": agree,
            "eye_gain_per_deg": round(float(g), 5), "frames_reading": int(mr.sum()), "notes": notes}


class NullFinder:
    """Online driver: feed frames, read `state` for the UI, `result` when `done`."""

    def __init__(self, t0: float, dot_x: dict):
        self.t0, self.dot_x, self.rows = t0, dict(dot_x), []
        self.done, self.result = False, None
        self.state = {}
        self.step(t0, None)

    def step(self, t: float, g: np.ndarray | None):
        el = t - self.t0
        s = schedule(el)
        if s is None:
            if not self.done:
                self.done, self.result = True, estimate(self.rows, self.dot_x)
            return self.state
        ph, key, i, into = s
        if g is not None:
            self.rows.append((ph, key, into, el, float((g[0] + g[3]) / 2), float(g[9])))
        st = {"phase": ph, "i": i, "n": len(STEPS), "p": round(min(1.0, into / (WORD_S if ph == "read" else DOT_S)), 2),
              "total_p": round(min(1.0, el / total_s()), 3)}
        if ph == "read":
            st.update(word=WORDS[key], size=SIZES[key])
        else:
            st.update(x=self.dot_x[key], zone=key)
        self.state = st
        return st
