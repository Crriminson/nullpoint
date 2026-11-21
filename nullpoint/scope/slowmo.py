"""Track the iris in close-up slow-motion phone video (no face detector needed).

Output: per-frame eye-in-head horizontal/vertical position in degrees.

Method (classical CV, chosen because MediaPipe finds no face in eye close-ups)
1. Initial iris: Hough circles (gradient-alt), each candidate scored by contrast between a dark
   interior and a brighter annulus (sclera). Best-scoring circle wins.
2. Per frame, limbus fit ("starburst"): cast rays from the previous centre in two horizontal
   fans (left/right +-50 deg, where the sclera is visible and the eyelids are not), take the
   strongest dark->bright gradient along each ray between 0.6r and 1.6r, then fit a circle to
   these edge points with RANSAC + least squares -> sub-pixel centre and radius.
3. Head/camera reference: a textured patch above the eye (brow/skin), tracked with NCC
   (sub-pixel). Eye-in-head = iris centre - patch displacement.
4. px -> deg: horizontal iris diameter = 11.7 mm, rotation radius 12 mm: deg = asin(dx_mm/12).
   The radius is re-measured every frame, so the phone moving closer/further is handled.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from nullpoint.config import IRIS_DIAMETER_MM, EYE_ROTATION_RADIUS_MM

PROC_W = 480


@dataclass
class Track:
    fps: float
    t: np.ndarray            # s
    iris_px: np.ndarray      # (N,2) iris centre, processed-frame px (NaN = lost)
    radius_px: np.ndarray    # (N,) fitted limbus radius
    ref_px: np.ndarray       # (N,2) head/camera reference displacement
    score: np.ndarray        # (N,) fit quality: inlier fraction

    @property
    def mm_per_px(self) -> np.ndarray:
        r = _smooth_nan(self.radius_px, 51)
        return IRIS_DIAMETER_MM / (2 * r)

    @property
    def segments(self) -> list[slice]:
        """Contiguous tracked stretches (split at template re-cuts and losses)."""
        cut = np.flatnonzero((self.score == -1) | ~np.isfinite(self.iris_px[:, 0]))
        edges = np.r_[0, cut, len(self.t)]
        return [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b - a > 1]

    def eye_deg(self, head_comp: bool = True) -> np.ndarray:
        """(N,2) eye-in-head angle in degrees (x positive = image right), zero-median per segment."""
        from scipy.signal import medfilt
        p = self.iris_px - (self.ref_px if head_comp else 0)
        out = np.full_like(p, np.nan)
        for sg in self.segments:
            q = p[sg].copy()
            for k in range(2):
                c = q[:, k]; good = np.isfinite(c)
                if good.sum() < 5:
                    continue
                c[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), c[good])
                q[:, k] = medfilt(c, 3) - np.median(c)
            out[sg] = q
        mm = out * self.mm_per_px[:, None]
        return np.degrees(np.arcsin(np.clip(mm / EYE_ROTATION_RADIUS_MM, -1, 1)))

    def save(self, path: Path):
        np.savez_compressed(path, fps=self.fps, t=self.t, iris_px=self.iris_px,
                            radius_px=self.radius_px, ref_px=self.ref_px, score=self.score)

    @classmethod
    def load(cls, path: Path) -> "Track":
        d = np.load(path)
        return cls(float(d["fps"]), d["t"], d["iris_px"], d["radius_px"], d["ref_px"], d["score"])


def _smooth_nan(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, float).copy()
    good = np.isfinite(x)
    if good.sum() == 0:
        return x
    x[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), x[good])
    k = max(1, min(k, len(x)) | 1)
    return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def gray(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    s = PROC_W / w
    g = cv2.resize(frame, (PROC_W, int(round(h * s))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6) * 45 + 128      # cancel slow-mo light flicker
    return np.clip(g, 0, 255).astype(np.uint8)


def _circle_contrast(g: np.ndarray, x: float, y: float, r: float) -> float:
    """Mean(annulus 1.15r..1.6r, horizontal sectors) - mean(disk 0..0.85r). Higher = more iris-like."""
    h, w = g.shape
    R = int(1.7 * r) + 1
    x0, y0, x1, y1 = int(x) - R, int(y) - R, int(x) + R + 1, int(y) + R + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return -1e9
    patch = g[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.hypot(xx - x, yy - y)
    ang = np.abs(np.degrees(np.arctan2(yy - y, xx - x)))
    horiz = (ang < 45) | (ang > 135)
    inner = patch[d < 0.85 * r]
    outer = patch[(d > 1.15 * r) & (d < 1.6 * r) & horiz]
    if inner.size < 10 or outer.size < 10:
        return -1e9
    return float(np.median(outer) - np.median(inner))


def detect_iris(g: np.ndarray, rmin: int, rmax: int, near=None, near_tol=None):
    """Return (x, y, r, contrast) of the most iris-like circle, or None."""
    gb = cv2.GaussianBlur(g, (0, 0), 2)
    cands = []
    for p2 in (0.6, 0.4, 0.25):
        c = cv2.HoughCircles(gb, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=rmin, param1=60, param2=p2,
                             minRadius=rmin, maxRadius=rmax)
        if c is not None:
            cands += [tuple(v) for v in c[0]]
        if len(cands) >= 3:
            break
    best = None
    dark_thr = np.percentile(gb, 25)
    for x, y, r in cands:
        if near is not None and np.hypot(x - near[0], y - near[1]) > near_tol:
            continue
        yy, xx = np.ogrid[:gb.shape[0], :gb.shape[1]]
        inner = gb[(xx - x) ** 2 + (yy - y) ** 2 < (0.6 * r) ** 2]
        if inner.size == 0 or np.median(inner) > dark_thr:   # the iris/pupil must be among the darkest pixels
            continue
        s = _circle_contrast(gb, x, y, r)
        if best is None or s > best[3]:
            best = (float(x), float(y), float(r), s)
    return best


_ANGLES = np.radians(np.r_[np.linspace(-50, 50, 15), np.linspace(130, 230, 15)])


def limbus_fit(g: np.ndarray, x: float, y: float, r: float, rng=np.random.default_rng(0)):
    """Starburst limbus fit. Returns (x, y, r, inlier_frac) or None."""
    gb = cv2.GaussianBlur(g, (0, 0), 1.2).astype(np.float32)
    h, w = gb.shape
    s = np.linspace(0.6 * r, 1.6 * r, int(1.0 * r * 2) + 2)
    pts = []
    for a in _ANGLES:
        xs, ys = x + s * np.cos(a), y + s * np.sin(a)
        ok = (xs >= 1) & (xs < w - 2) & (ys >= 1) & (ys < h - 2)
        if ok.sum() < 5:
            continue
        prof = cv2.remap(gb, xs[ok].astype(np.float32)[None], ys[ok].astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        d = np.gradient(prof)
        k = int(np.argmax(d))
        if d[k] < 2.0:             # no clear dark->bright edge
            continue
        # sub-sample edge position (parabola on gradient)
        off = 0.0
        if 0 < k < len(d) - 1:
            den = d[k - 1] - 2 * d[k] + d[k + 1]
            off = 0.5 * (d[k - 1] - d[k + 1]) / den if den != 0 else 0.0
        ss = s[ok]
        si = np.interp(k + np.clip(off, -1, 1), np.arange(len(ss)), ss)
        pts.append((x + si * np.cos(a), y + si * np.sin(a)))
    if len(pts) < 6:
        return None
    P = np.array(pts)

    def fit_center(Q, R):  # geometric fit of the centre with known radius (Gauss-Newton)
        c = Q.mean(0) if len(Q) else np.array([x, y])
        c = np.array([x, y], float)
        for _ in range(8):
            d = Q - c; dist = np.hypot(d[:, 0], d[:, 1]) + 1e-9
            res = dist - R
            J = -d / dist[:, None]
            step, *_ = np.linalg.lstsq(J, -res, rcond=None)
            c = c + step
            if np.abs(step).max() < 1e-3:
                break
        return c

    R = r
    best_in = None
    for _ in range(40):
        idx = rng.choice(len(P), 2, replace=False)
        c = fit_center(P[idx], R)
        if np.hypot(c[0] - x, c[1] - y) > 0.8 * R:
            continue
        inl = np.abs(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) - R) < max(1.0, 0.07 * R)
        if best_in is None or inl.sum() > best_in.sum():
            best_in = inl
    if best_in is None or best_in.sum() < 6:
        return None
    c = fit_center(P[best_in], R)
    # free-radius estimate (only used to slowly adapt R when both limbus sides are visible)
    Q = P[best_in]; r_free = np.nan
    if min((Q[:, 0] < c[0]).sum(), (Q[:, 0] >= c[0]).sum()) >= 3:
        A = np.c_[2 * Q[:, 0], 2 * Q[:, 1], np.ones(len(Q))]; b = (Q ** 2).sum(1)
        k, *_ = np.linalg.lstsq(A, b, rcond=None)
        r_free = float(np.sqrt(max(k[2] + k[0] ** 2 + k[1] ** 2, 1e-6)))
    return float(c[0]), float(c[1]), r_free, float(best_in.mean())


class RefPatch:
    """Head/camera reference: NCC-tracked textured patch above the eye."""

    def __init__(self, g, x, y, r):
        self.w, self.h = int(3.0 * r), int(1.2 * r)
        self.x0, self.y0 = int(x - self.w / 2), int(y - 2.6 * r - self.h / 2)
        self.ok = self.y0 >= 0 and self.x0 >= 0 and self.x0 + self.w < g.shape[1]
        if not self.ok:   # not enough room above: use the region below the eye
            self.y0 = int(y + 1.8 * r)
            self.ok = self.y0 + self.h < g.shape[0] and self.x0 >= 0 and self.x0 + self.w < g.shape[1]
        if self.ok:
            self.tpl = self._prep(g[self.y0:self.y0 + self.h, self.x0:self.x0 + self.w])
        self.pos = (0.0, 0.0)

    @staticmethod
    def _prep(p):
        return cv2.GaussianBlur(p, (0, 0), 1.0).astype(np.float32)

    def update(self, g, search):
        if not self.ok:
            return (np.nan, np.nan), 0.0
        sx, sy = int(search), int(search)
        px, py = self.x0 + self.pos[0], self.y0 + self.pos[1]
        X0, Y0 = int(max(px - sx, 0)), int(max(py - sy, 0))
        X1, Y1 = int(min(px + self.w + sx, g.shape[1])), int(min(py + self.h + sy, g.shape[0]))
        roi = self._prep(g[Y0:Y1, X0:X1])
        if roi.shape[0] <= self.h or roi.shape[1] <= self.w:
            return (np.nan, np.nan), 0.0
        res = cv2.matchTemplate(roi, self.tpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, (ix, iy) = cv2.minMaxLoc(res)
        dx = dy = 0.0
        if 0 < ix < res.shape[1] - 1:
            l, c, r_ = res[iy, ix - 1:ix + 2]; den = l - 2 * c + r_
            dx = 0.5 * (l - r_) / den if den else 0.0
        if 0 < iy < res.shape[0] - 1:
            u, c, d = res[iy - 1:iy + 2, ix]; den = u - 2 * c + d
            dy = 0.5 * (u - d) / den if den else 0.0
        self.pos = (X0 + ix + dx - self.x0, Y0 + iy + dy - self.y0)
        return self.pos, float(mx)


def _ncc_peak(img, tpl):
    res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, (ix, iy) = cv2.minMaxLoc(res)
    dx = dy = 0.0
    if 0 < ix < res.shape[1] - 1:
        l, c, r_ = res[iy, ix - 1:ix + 2]; den = l - 2 * c + r_
        dx = 0.5 * (l - r_) / den if den else 0.0
    if 0 < iy < res.shape[0] - 1:
        u, c, d = res[iy - 1:iy + 2, ix]; den = u - 2 * c + d
        dy = 0.5 * (u - d) / den if den else 0.0
    return ix + float(np.clip(dx, -1, 1)), iy + float(np.clip(dy, -1, 1)), float(mx)


class IrisTemplate:
    """Appearance template of the iris (+ limbus, a little sclera) cut from a detected frame."""

    def __init__(self, g, x, y, r):
        self.hw, self.hh = int(1.35 * r), int(0.7 * r)   # wide, short: limbus edges, avoids lids
        self.tpl = cv2.GaussianBlur(g, (0, 0), 1.0)[int(y) - self.hh:int(y) + self.hh + 1,
                                                    int(x) - self.hw:int(x) + self.hw + 1].astype(np.float32)
        self.off = (x - int(x), y - int(y))
        self.ok = self.tpl.shape == (2 * self.hh + 1, 2 * self.hw + 1)

    def locate(self, g, x, y, search):
        gb = cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.float32)
        X0, Y0 = int(max(x - self.hw - search, 0)), int(max(y - self.hh - search, 0))
        X1, Y1 = int(min(x + self.hw + search + 1, g.shape[1])), int(min(y + self.hh + search + 1, g.shape[0]))
        roi = gb[Y0:Y1, X0:X1]
        if roi.shape[0] <= self.tpl.shape[0] or roi.shape[1] <= self.tpl.shape[1]:
            return None
        ix, iy, sc = _ncc_peak(roi, self.tpl)
        return X0 + ix + self.hw + self.off[0], Y0 + iy + self.hh + self.off[1], sc


def track_video(path: Path, max_frames: int | None = None, verbose=True, min_score=0.6,
                init: tuple[float, float, float] | None = None) -> Track:
    """NCC appearance tracking of the iris + a brow/skin reference patch.

    Segments: whenever the iris template stops matching (blink, big gaze change, refocus),
    the iris is re-detected and both templates are re-cut. Positions are made continuous
    across a re-cut by carrying the offset of the first frame after it (a jump marker is kept
    in `score` = -1 so analysis can split there)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = PROC_W
    rmin, rmax = int(0.03 * W), int(0.12 * W)

    def detect(g, near=None, tol=None, rr=None):
        lo, hi = (int(0.8 * rr), int(1.25 * rr) + 1) if rr else (rmin, rmax)
        d = detect_iris(g, lo, hi, near, tol)
        if d is None:
            return None
        fr = limbus_fit(g, d[0], d[1], d[2])
        return (fr[0], fr[1], d[2]) if fr else d[:3]

    frames0 = []
    for _ in range(30):
        ok, f = cap.read()
        if not ok:
            break
        frames0.append(gray(f))
    if init is None:
        cands = [c for c in (detect_iris(g, rmin, rmax) for g in frames0) if c]
        if not cands:
            raise RuntimeError("no iris found in first frames; pass --init x,y,r or --click")
        best = max(cands, key=lambda c: c[3])
        init = best[:3]
    r_ref = float(init[2])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    iris, refs, score, rad = [], [], [], []
    g0 = frames0[0]
    x, y = float(init[0]), float(init[1])
    it = IrisTemplate(g0, x, y, r_ref); it = it if it.ok else None
    ref = RefPatch(g0, x, y, r_ref)
    iris_base = ref_base = np.zeros(2)
    i = 0
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        g = gray(f)
        res = it.locate(g, x, y, search=0.6 * r_ref) if it else None
        if res is not None and res[2] >= min_score:
            x, y, sc = res
            (px, py), rq = ref.update(g, search=1.0 * r_ref) if ref and ref.ok else ((np.nan, np.nan), 0)
            iris.append((x, y)); score.append(sc); rad.append(r_ref)
            refs.append((px, py) if rq > 0.5 else (np.nan, np.nan))
        else:
            d = detect(g, near=(x, y) if x is not None else None, tol=3 * r_ref if x is not None else None, rr=r_ref)
            if d is None and x is not None:
                d = detect(g, rr=r_ref)
            if d is not None:
                x, y = d[0], d[1]
                it = IrisTemplate(g, x, y, r_ref)
                if not it.ok:
                    it = None
                ref = RefPatch(g, x, y, r_ref)
                iris.append((x, y)); score.append(-1.0); rad.append(r_ref); refs.append((0.0, 0.0) if ref.ok else (np.nan, np.nan))
            else:
                iris.append((np.nan, np.nan)); score.append(0.0); rad.append(np.nan); refs.append((np.nan, np.nan))
        i += 1
        if verbose and i % 1000 == 0:
            sc = np.array(score)
            print(f"  {i} frames, tracked {np.mean(sc > 0):.0%}, re-cuts {np.sum(sc == -1)}")
    n = len(iris)
    return Track(fps, np.arange(n) / fps, np.array(iris, float), np.array(rad, float),
                 np.array(refs, float), np.array(score, float))


def click_init(video: Path) -> tuple[float, float, float]:
    """Operator marks the iris: click its centre, then a point on the limbus (iris edge)."""
    ok, f = cv2.VideoCapture(str(video)).read()
    g = cv2.cvtColor(gray(f), cv2.COLOR_GRAY2BGR); pts = []
    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y)); cv2.circle(g, (x, y), 3, (0, 0, 255), -1); cv2.imshow("mark iris", g)
    cv2.imshow("mark iris", g); cv2.setMouseCallback("mark iris", cb)
    while len(pts) < 2 and cv2.waitKey(30) != 27:
        pass
    cv2.destroyAllWindows()
    (cx, cy), (ex, ey) = pts
    return float(cx), float(cy), float(np.hypot(ex - cx, ey - cy))


def main():
    ap = argparse.ArgumentParser(description="Track iris in close-up slow-mo video")
    ap.add_argument("video", type=Path)
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--init", help="x,y,r of the iris in the first frame (processed-frame px, width 480)")
    ap.add_argument("--click", action="store_true", help="click iris centre then limbus edge on the first frame")
    a = ap.parse_args()
    init = tuple(float(v) for v in a.init.split(",")) if a.init else None
    if a.click:
        init = click_init(a.video)
    tr = track_video(a.video, a.max_frames, init=init)
    out = a.out or a.video.with_suffix(".track.npz")
    tr.save(out)
    valid = np.isfinite(tr.iris_px[:, 0])
    print(json.dumps({"fps": round(tr.fps, 2), "frames": len(tr.t), "valid": round(float(valid.mean()), 3),
                      "ref_valid": round(float(np.isfinite(tr.ref_px[:, 0]).mean()), 3),
                      "radius_px_median": round(float(np.nanmedian(tr.radius_px)), 1),
                      "px_per_deg": round(float(np.nanmedian(2 * tr.radius_px) / IRIS_DIAMETER_MM * np.radians(1) * EYE_ROTATION_RADIUS_MM), 2),
                      "out": str(out)}, indent=1))


if __name__ == "__main__":
    main()
