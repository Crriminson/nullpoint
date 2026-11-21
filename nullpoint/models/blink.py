"""Deliberate-vs-natural blink classification.

Event detection (online): closure score s = mean(eyeBlinkLeft, eyeBlinkRight); an event starts
when s > ON and ends when s < OFF (hysteresis); events closer than 80 ms are merged.
Event features: duration, peak/mean closure, area, rise/fall times and velocities, L/R
asymmetry, squint, brow, minimum lid opening.

Classifiers compared in eval/blink_eval.py:
  * DurationThreshold : baseline used by most blink boards (duration > tau)
  * LightGBM          : on the event features
  * GRU               : on the raw per-frame sequence around the event (sequence model)
Emergency (a deterministic rule, not a model): close the eyes for EMERGENCY_S..EMERGENCY_MAX_S
seconds, then open them. At EMERGENCY_S the board gets 'emergency_pending' (a soft cue: "open your
eyes to call for help"); the alarm fires on reopening. Longer closures are rest or sleep and end as
'rest', so dozing off never calls the caregiver.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np

ON, OFF = 0.45, 0.30
EMERGENCY_S = 2.0
EMERGENCY_MAX_S = 8.0
SEQ_LEN = 40                  # frames (~1.3 s at 30 fps) for the GRU, from event start - 4 frames
EVENT_FEATS = ["dur", "peak", "mean", "area", "rise_t", "fall_t", "rise_v", "fall_v", "asym", "squint",
               "brow", "min_open", "pre_open"]


@dataclass
class BlinkEvent:
    i0: int
    i1: int          # exclusive
    t0: float
    t1: float
    feats: np.ndarray
    seq: np.ndarray  # (SEQ_LEN, 6)


def _seq(B, i0, i1, n=SEQ_LEN):
    """Window from 4 frames before the event to its end (what is known when it finishes),
    truncated/padded to n frames, so training and live inference see the same thing."""
    a = max(0, i0 - 4)
    s = B[a:min(i1 + 1, a + n)][:, [0, 1, 2, 3, 4, 5]]
    if len(s) < n:
        s = np.vstack([s, np.repeat(s[-1:], n - len(s), 0)])
    return np.nan_to_num(s.astype(np.float32))


def event_features(t, B, i0, i1) -> np.ndarray:
    seg = B[i0:i1]
    s = (seg[:, 0] + seg[:, 1]) / 2
    ts = t[i0:i1]
    dur = float(t[i1 - 1] - t[i0]) + float(np.median(np.diff(t))) if i1 - i0 > 1 else float(np.median(np.diff(t)))
    k = int(np.argmax(s))
    v = np.gradient(s, ts) if len(s) > 2 else np.zeros_like(s)
    pre = B[max(0, i0 - 10):i0]
    pre_open = float(np.nanmean(pre[:, 4:6])) if len(pre) else float(np.nanmean(seg[:, 4:6]))
    return np.array([dur, s.max(), s.mean(), s.sum() * dur / max(len(s), 1), ts[k] - ts[0], ts[-1] - ts[k],
                     v.max() if len(v) else 0, -v.min() if len(v) else 0,
                     float(np.mean(np.abs(seg[:, 0] - seg[:, 1]))), float(np.mean(seg[:, 2:4])),
                     float(np.mean(seg[:, 6:8])), float(np.nanmin(seg[:, 4:6])), pre_open], np.float32)


def detect_events(t, B) -> list[BlinkEvent]:
    s = np.nan_to_num((B[:, 0] + B[:, 1]) / 2)
    ev, on, i0 = [], False, 0
    for i, v in enumerate(s):
        if not on and v > ON:
            on, i0 = True, i
        elif on and v < OFF:
            on = False
            ev.append([i0, i])
    if on:
        ev.append([i0, len(s)])
    merged = []
    for a, b in ev:
        if merged and t[a] - t[merged[-1][1] - 1] < 0.08:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [BlinkEvent(a, b, float(t[a]), float(t[b - 1]), event_features(t, B, a, b), _seq(B, a, b)) for a, b in merged if b - a >= 1]


def label_events(sess, events) -> np.ndarray:
    """1 = deliberate (started within 1.8 s after a BLINK cue), 0 = natural, -1 = ambiguous."""
    cue_t = sess.t[sess.cue > 0]
    y = []
    for e in events:
        blk = sess.block[e.i0]
        if blk == "blink":
            after = e.t0 - cue_t
            y.append(1 if np.any((after >= 0.0) & (after <= 1.8)) else 0)
        elif blk in ("natural", "glance"):
            y.append(0)
        else:
            y.append(-1)
    return np.array(y)


class DurationThreshold:
    def fit(self, X, y):
        d = X[:, 0]
        cands = np.unique(d)
        best = max(cands, key=lambda c: (((d >= c) == (y == 1)).mean(), -c))
        self.tau = float(best); return self

    def predict_proba(self, X):
        p = (X[:, 0] >= self.tau).astype(float)
        return np.c_[1 - p, p]


def rule_model(tau: float = 0.45) -> "BlinkModel":
    """Blink model that needs no training data: closures lasting >= tau seconds are deliberate
    (natural blinks last ~0.1-0.3 s). Used until the user records a calibration session."""
    thr = DurationThreshold(); thr.tau = tau
    return BlinkModel(thr, "threshold")


def make_lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=4,
                          class_weight="balanced", verbose=-1)


class GRUBlink:
    """Small GRU over the per-frame blink signals around an event."""

    def __init__(self, hidden=24, epochs=60, seed=0):
        self.hidden, self.epochs, self.seed = hidden, epochs, seed

    def _net(self):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(s, h):
                super().__init__(); s.g = nn.GRU(6, h, batch_first=True); s.o = nn.Linear(h, 2)
            def forward(s, x):
                _, hN = s.g(x); return s.o(hN[-1])
        return Net(self.hidden)

    def fit(self, S, y):
        import torch
        torch.manual_seed(self.seed)
        self.mu, self.sd = S.reshape(-1, S.shape[-1]).mean(0), S.reshape(-1, S.shape[-1]).std(0) + 1e-6
        X = torch.tensor((S - self.mu) / self.sd, dtype=torch.float32); Y = torch.tensor(y)
        w = torch.tensor([1.0, max(1.0, (y == 0).sum() / max((y == 1).sum(), 1))], dtype=torch.float32)
        self.net = self._net(); opt = torch.optim.Adam(self.net.parameters(), 1e-2)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        for _ in range(self.epochs):
            opt.zero_grad(); l = lossf(self.net(X), Y); l.backward(); opt.step()
        self.net.eval(); return self

    def predict_proba(self, S):
        import torch
        with torch.no_grad():
            return torch.softmax(self.net(torch.tensor((S - self.mu) / self.sd, dtype=torch.float32)), 1).numpy()


class BlinkModel:
    """Online wrapper used by the board: feed frames, get finished events with p(deliberate)."""

    def __init__(self, clf=None, kind="lgbm", threshold=0.5):
        self.clf, self.kind, self.threshold = clf, kind, threshold
        self.reset()

    def reset(self):
        self.on, self.buf_t, self.buf_B, self.closed_since = False, [], [], None
        self.armed = False                     # closure has passed EMERGENCY_S

    def step(self, t: float, b: np.ndarray | None):
        """Returns ('deliberate'|'natural'|'emergency_pending'|'emergency'|'rest'|None, info)."""
        if b is None:
            return None, {}
        s = (b[0] + b[1]) / 2
        self.buf_t.append(t); self.buf_B.append(b)
        if len(self.buf_t) > 240:
            self.buf_t.pop(0); self.buf_B.pop(0)
            if self.on:
                self.i0 = max(0, self.i0 - 1)
        if not self.on and s > ON:
            self.on, self.i0 = True, len(self.buf_t) - 1
            self.closed_since = t
        elif self.on:
            if s < OFF:
                self.on = False
                armed, self.armed = self.armed, False
                if armed:                          # a long closure just ended: alarm, or just rest
                    dur = float(t - self.closed_since)
                    return ("emergency" if dur <= EMERGENCY_MAX_S else "rest"), {"dur": dur}
                T, B = np.array(self.buf_t), np.array(self.buf_B)
                i0 = self.i0
                f = event_features(T, B, i0, len(T) - 1)
                if self.kind == "gru":
                    p = float(self.clf.predict_proba(_seq(B, i0, len(B) - 1)[None])[0, 1])
                else:
                    p = float(self.clf.predict_proba(f[None])[0, 1])
                return ("deliberate" if p >= self.threshold else "natural"), {"p": p, "dur": float(f[0])}
            if not self.armed and t - self.closed_since >= EMERGENCY_S:
                self.armed = True
                return "emergency_pending", {"dur": t - self.closed_since}
        return None, {}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "kind": self.kind, "threshold": self.threshold}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["clf"], d["kind"], d["threshold"])
