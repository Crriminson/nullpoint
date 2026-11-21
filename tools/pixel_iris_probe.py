"""Pixel-level iris motion probe (bypasses MediaPipe's iris smoothing).

For each target segment: fixed eye ROI (from median landmarks), upsampled; per-frame
iris x = centroid of darkest pixels (subpixel), minus the ROI's own shift measured by
phase correlation on a skin patch (head-motion compensation). Also a montage of eye crops.
"""
import sys, pathlib
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend

npz = pathlib.Path(sys.argv[1]); d = np.load(npz)
cap = cv2.VideoCapture(str(npz.with_suffix(".mp4")))
frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
n = min(len(frames), len(d["t"])); frames = np.stack(frames[:n])
t, tgt = d["t"][:n], d["target"][:n]; L = d["lms"][:n].copy(); W, H = d["frame_wh"]; L[..., 0] *= W; L[..., 1] *= H
fs = 1 / np.median(np.diff(t))
EYES = {"R": (33, 133, 468), "L": (362, 263, 473)}
UP = 4

def iris_x(roi):
    r = cv2.resize(roi, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    r = cv2.GaussianBlur(r, (0, 0), 2)
    thr = np.percentile(r, 12)
    w = np.clip(thr - r, 0, None)           # darkness weight
    cols = np.arange(r.shape[1])
    return (w.sum(0) @ cols) / max(w.sum(), 1e-6) / UP

fig, ax = plt.subplots(3, 1, figsize=(14, 10)); res = []
for x in dict.fromkeys(tgt):
    m = np.where((tgt == x) & (t > t[tgt == x][0] + 1.0))[0]
    for name, (a, b, ir) in EYES.items():
        cx, cy = np.median(L[m, ir, 0]), np.median(L[m, ir, 1])
        ew = np.median(np.abs(L[m, b, 0] - L[m, a, 0]))
        x0, x1, y0, y1 = int(cx - .6 * ew), int(cx + .6 * ew), int(cy - .25 * ew), int(cy + .25 * ew)
        # skin reference patch just below the eye for head-motion compensation
        ref0 = frames[m[0], y1 + 5:y1 + 25, x0:x1].astype(np.float32)
        xs, hs = [], []
        for i in m:
            xs.append(iris_x(frames[i, y0:y1, x0:x1]))
            sh, _ = cv2.phaseCorrelate(ref0, frames[i, y1 + 5:y1 + 25, x0:x1].astype(np.float32))
            hs.append(sh[0])
        s = np.array(xs) - np.array(hs)
        fr, P = welch(detrend(s), fs=fs, nperseg=min(128, len(s)))
        band = (fr >= 1) & (fr <= 8); k = np.argmax(np.where(band, P, 0))
        snr = P[k] / np.median(P[fr > 8])
        p2p = np.percentile(s, 95) - np.percentile(s, 5)
        res.append((x, name, fr[k], snr, p2p, ew))
        print(f"target {x:.1f} {name}: peak {fr[k]:.2f} Hz  peak/HF {snr:7.1f}  p2p {p2p:.2f}px (eye width {ew:.0f}px, iris~{ew*0.45:.0f}px)  head-shift sd {np.std(hs):.2f}px")
        ax[1].semilogy(fr, P, lw=.8, label=f"{x:.1f}{name}")
        if x == tgt[0]:
            ax[0].plot(t[m], s - s.mean(), ".-", ms=3, lw=.8, label=f"{name} iris-x minus head shift (px)")
            ax[2].plot(t[m], np.array(hs) - np.mean(hs), lw=.8, label=f"{name} head shift (px)")
            if name == "R":
                crops = [cv2.resize(frames[i, y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC) for i in m[30:54]]
                rows = [np.hstack(crops[j:j + 6]) for j in range(0, 24, 6)]
                cv2.imwrite(str(npz.with_name(npz.stem + "-eyecrops.png")), np.vstack(rows))
for a_ in ax: a_.legend(fontsize=7, ncol=5)
ax[1].set_xlabel("Hz"); plt.tight_layout(); plt.savefig(npz.with_name(npz.stem + "-pixel.png"), dpi=90)
