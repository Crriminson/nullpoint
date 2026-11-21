"""Glance-zone model: per-frame classifier (L / C / R) + HMM smoothing.

Per-frame classifier: LDA (fast, works with few samples) or LightGBM, trained on the user's own
calibration frames after a settle time. Baseline: a threshold on the mean horizontal iris
position (what a non-learned board would do).

Temporal model: a 3-state HMM whose emissions are the classifier's posteriors and whose
transition matrix makes zone switches rare (sticky). Forward filtering gives an online,
causal posterior; it suppresses frame-level flicker that nystagmus or landmark noise causes.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ZONES = ["L", "C", "R"]

# Feature sets (columns of features.GAZE_NAMES). "eye" = eye-in-head only: iris position and lid
# opening of both eyes, plus the gaze blendshapes. Head position/pose ("all") separated zones within
# a session but shifted between sessions (nose position by up to 3 SD in two test sessions), so
# a model that used it made phantom steps the next session (eval/recal.py). Default: "eye".
FEATURE_SETS = {"eye": [0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16], "eye_geom": [0, 1, 2, 3, 4, 5], "all": None}


def glance_frames(sess, settle_s: float = 0.6):
    """Frames from the glance block after the settle time, with zone labels."""
    m = (sess.block == "glance") & np.isfinite(sess.G).all(1)
    idx = np.flatnonzero(m)
    keep = []
    start_t, prev = None, None
    for i in idx:
        if sess.target[i] != prev or (keep and sess.t[i] - sess.t[keep[-1]] > 0.5):
            start_t, prev = sess.t[i], sess.target[i]
        if sess.t[i] - start_t >= settle_s:
            keep.append(i)
    keep = np.array(keep, int)
    y = np.array([ZONES.index(z) for z in sess.target[keep]])
    return sess.G[keep], y, keep


class ThresholdBaseline:
    """Non-learned baseline: mean iris x of both eyes vs two thresholds set halfway between
    zone means (still calibrated per user, but no model)."""
    def fit(self, G, y):
        x = (G[:, 0] + G[:, 3]) / 2
        mu = [np.median(x[y == k]) for k in range(3)]
        order = np.argsort(mu)
        self.sign = 1 if mu[2] > mu[0] else -1
        m = sorted(mu)
        self.t1, self.t2 = (m[0] + m[1]) / 2, (m[1] + m[2]) / 2
        self.map = order
        return self

    def predict_proba(self, G):
        x = (G[:, 0] + G[:, 3]) / 2
        k = np.where(x < self.t1, 0, np.where(x < self.t2, 1, 2))
        lab = self.map[k]
        P = np.full((len(G), 3), 0.02); P[np.arange(len(G)), lab] = 0.96
        return P

    def predict(self, G):
        return self.predict_proba(G).argmax(1)


def make_classifier(kind: str = "lda"):
    if kind == "lda":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10, verbose=-1)
    if kind == "threshold":
        return ThresholdBaseline()
    raise ValueError(kind)


@dataclass
class HMM:
    """Sticky 3-state HMM, forward filtering over classifier posteriors."""
    p_stay: float = 0.97
    belief: np.ndarray = field(default_factory=lambda: np.ones(3) / 3)

    def __post_init__(self):
        q = (1 - self.p_stay) / 2
        self.T = np.full((3, 3), q) + np.eye(3) * (self.p_stay - q)

    def reset(self):
        self.belief = np.ones(3) / 3

    def step(self, post: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
        prior = prior if prior is not None else np.ones(3) / 3
        lik = np.clip(post, 1e-4, 1) / prior          # posterior -> scaled likelihood
        b = (self.T.T @ self.belief) * lik
        self.belief = b / b.sum()
        return self.belief

    def filter(self, P: np.ndarray) -> np.ndarray:
        self.reset()
        return np.array([self.step(p) for p in P])


class GlanceModel:
    """Per-frame classifier + HMM, with few-shot adaptation in front of the classifier.

    `cols` restricts the features used (e.g. the quick-start model trained on old scans that have
    no blendshapes/pose uses only the landmark-geometry columns).

    Adaptation. The classifier is fitted once. A new session, or drift within one (the head moved,
    the laptop lid tilted), is handled by a per-feature affine map x' = a*x + b in front of it:
      * recalibrate(): from a few labelled seconds (look at 3-4 dots), choose b (and optionally a)
        so that the new session's zone means land on the training zone means. With the gain on,
        features that hardly change across zones (lid opening, nose position) can't identify it, so
        a ridge term holds a near 1. Nothing is retrained; it takes microseconds.
      * nudge(): self-correction from frames whose zone is known without asking (the focus card just
        before a deliberate blink). Moves only the offset b, part of the way, with a cap per update.
    """

    def __init__(self, kind="lda", p_stay=0.97, cols=None):
        self.kind, self.clf, self.hmm, self.cols = kind, make_classifier(kind), HMM(p_stay), cols
        self.mu = self.sd = self.a = self.b = None
        self.fitted = False
        self.zone_x = None          # screen x of the L / C / R positions it was trained on (for degrees)

    def _raw(self, G):
        G = np.atleast_2d(G)
        return G[:, self.cols] if self.cols is not None else G

    def _x(self, G):
        X = self._raw(G)
        return X * self.a + self.b if self.a is not None else X

    def fit(self, G, y):
        X = self._raw(G); y = np.asarray(y)
        self.clf.fit(X, y)
        self.mu = np.array([X[y == k].mean(0) for k in range(3)])
        self.sd = X.std(0) + 1e-6
        self.reset_alignment()
        self.fitted = True
        return self

    def proba(self, G):
        return self.clf.predict_proba(self._x(G))

    # ---------------------------------------------------------------- few-shot adaptation
    @property
    def adaptable(self):
        return self.mu is not None

    def reset_alignment(self):
        d = self.mu.shape[1]
        self.a, self.b = np.ones(d), np.zeros(d)

    def recalibrate(self, G, y, mode="offset", ridge=0.25, min_frames=5, keep_if=0.95):
        """Fit the alignment from labelled frames of the current session. Returns a summary.
        mode "offset" (default) shifts each feature; "affine" also fits a gain per feature. On the
        test recordings the offset generalised better to a zone the recalibration had not seen
        (eval/recal.py), so the gain is off by default.
        Do no harm: the alignment starts from the trained state, and if that already classifies at
        least `keep_if` of the setup frames correctly it is kept unchanged (None: always align)."""
        X = self._raw(G); y = np.asarray(y)
        ok = np.isfinite(X).all(1); X, y = X[ok], y[ok]
        ks = [k for k in range(3) if (y == k).sum() >= min_frames]
        if not ks:
            return {"ok": False, "reason": "too few frames"}
        self.reset_alignment()
        acc0 = float((self.proba(X).argmax(1) == y).mean())
        if keep_if is not None and acc0 >= keep_if:
            return {"ok": True, "changed": False, "zones": [ZONES[k] for k in ks], "frames": int(len(y)),
                    "acc_before": round(acc0, 3)}
        P = np.array([X[y == k].mean(0) for k in ks])            # this session's zone means
        w = np.array([(y == k).sum() for k in ks], float); w /= w.sum()
        c, s = self.mu.mean(0), self.sd
        p, q = (P - c) / s, (self.mu[ks] - c) / s                 # standardised by the training data
        Sp, Sq = (w[:, None] * p).sum(0), (w[:, None] * q).sum(0)
        if mode == "offset" or len(ks) < 2:
            a = np.ones_like(c)
        else:  # weighted least squares q ~ a p + beta, ridge pulling a towards 1
            Spp = (w[:, None] * p * p).sum(0) + ridge
            Spq = (w[:, None] * p * q).sum(0) + ridge
            a = np.clip((Spq - Sp * Sq) / (Spp - Sp * Sp), 0.5, 2.0)
        beta = Sq - a * Sp
        self.a, self.b = a, (1 - a) * c + s * beta
        moved = np.abs((P * self.a + self.b - P) / s)
        acc1 = float((self.proba(X).argmax(1) == y).mean())
        return {"ok": True, "changed": True, "zones": [ZONES[k] for k in ks], "frames": int(len(y)),
                "acc_before": round(acc0, 3), "acc_after": round(acc1, 3), "max_shift_sd": round(float(moved.max()), 2)}

    def nudge(self, G, k=1, rate=0.3, max_sd=0.5, min_frames=5, max_move=0.5, tol=0.15):
        """Self-correction from frames believed to be zone k. Moves the offset `rate` of the way to
        the training mean, capped at `max_sd` training SDs per feature. Measured on the most
        zone-separating feature, relative to the smallest gap between neighbouring zones, it is
        skipped (returns None)
          * if the eyes were moving: the frames' 10-90 % range is over `max_move` of the gap (a
            glance coming back from a side zone would teach the model that the side is the centre)
          * if there is nothing to correct: the frames' mean is within `tol` of the gap from the
            trained zone mean (do no harm)
        Else returns the largest step taken, in training SDs."""
        if not self.adaptable:
            return None
        X = self._x(G); X = X[np.isfinite(X).all(1)]
        if len(X) < min_frames:
            return None
        gap = np.diff(np.sort(self.mu, 0), axis=0).min(0)
        j = int(np.argmax(gap / self.sd))
        lo, hi = np.percentile(X[:, j], [10, 90])
        if hi - lo > max_move * gap[j]:
            return None
        if abs(X[:, j].mean() - self.mu[k, j]) < tol * gap[j]:
            return None
        step = np.clip(rate * (self.mu[k] - X.mean(0)), -max_sd * self.sd, max_sd * self.sd)
        self.b = self.b + step
        return float(np.abs(step / self.sd).max())

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"kind": self.kind, "clf": self.clf, "p_stay": self.hmm.p_stay, "cols": self.cols,
                         "mu": self.mu, "sd": self.sd, "a": self.a, "b": self.b, "zone_x": self.zone_x}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls(d["kind"], d["p_stay"], d.get("cols")); m.clf = d["clf"]
        m.mu, m.sd, m.a, m.b = d.get("mu"), d.get("sd"), d.get("a"), d.get("b")
        m.zone_x = d.get("zone_x")
        m.fitted = True
        return m
