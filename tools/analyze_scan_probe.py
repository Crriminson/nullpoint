"""Analyse a gaze-scan probe: is the nystagmus visible in webcam landmarks?

Per target segment: horizontal iris signal (landmark-based, two references), Welch PSD peak,
peak-to-peak amplitude. Writes a diagnostic PNG next to the npz.
"""
import sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend

f = pathlib.Path(sys.argv[1]); d = np.load(f)
t, tgt, L = d["t"], d["target"], d["lms"].copy()
W, H = d["frame_wh"]
L[..., 0] *= W; L[..., 1] *= H
fs = 1 / np.median(np.diff(t))
print(f"frames {len(t)}, fs {fs:.1f} Hz, face found {np.isfinite(L[:,0,0]).mean():.0%}")

def eye_signal(c_out, c_in, iris, ring):
    a, b = L[:, c_out, :2], L[:, c_in, :2]
    u = (b - a) / np.linalg.norm(b - a, axis=1, keepdims=True)
    width = np.linalg.norm(b - a, axis=1)
    x_corner = np.einsum("ij,ij->i", L[:, iris, :2] - a, u) / width     # 0..1 across the eye
    iris_d = np.linalg.norm(L[:, ring[0], :2] - L[:, ring[1], :2], axis=1)
    return x_corner, iris_d

# MediaPipe: subject's right eye corners 33 (outer) / 133 (inner), iris 468, ring 469/471
#            subject's left eye 263 (outer) / 362 (inner), iris 473, ring 474/476
xr, dr = eye_signal(33, 133, 468, (469, 471))
xl, dl = eye_signal(362, 263, 473, (474, 476))
# head-referenced: iris x minus midpoint of the 4 eye corners, in iris diameters
mid = L[:, [33, 133, 362, 263], 0].mean(1)
hr = (L[:, 468, 0] - mid) / np.nanmedian(dr)
hl = (L[:, 473, 0] - mid) / np.nanmedian(dl)
print(f"iris diameter px R {np.nanmedian(dr):.1f} L {np.nanmedian(dl):.1f}")

fig, ax = plt.subplots(3, 1, figsize=(14, 10))
ax[0].plot(t, xr, lw=.8, label="R eye (corner-norm)"); ax[0].plot(t, xl, lw=.8, label="L eye (corner-norm)")
ax[0].plot(t, d["blinkL"], lw=.6, c="k", alpha=.4, label="blink score"); ax[0].legend(); ax[0].set_title(f.name)
segs = []
for k, x in enumerate(dict.fromkeys(tgt)):
    m = (tgt == x); ts = t[m]; keep = m & (t > ts[0] + 1.0)   # skip the 1 s saccade/settle
    for name, sig in (("R", xr), ("L", xl)):
        s = sig[keep]; s = s[np.isfinite(s)]
        if len(s) < 64: continue
        fr, P = welch(detrend(s), fs=fs, nperseg=min(128, len(s)))
        band = (fr >= 1) & (fr <= 8); kk = np.argmax(np.where(band, P, 0))
        snr = P[kk] / np.median(P[(fr > 8)]) if (fr > 8).any() else np.nan
        p2p = np.percentile(s, 95) - np.percentile(s, 5)
        segs.append((x, name, np.mean(s), fr[kk], snr, p2p))
        print(f"target {x:.1f} eye {name}: mean {np.mean(s):.3f}  peak {fr[kk]:.2f} Hz  peak/HF-noise {snr:6.1f}  p2p(5-95%) {p2p:.3f} eye-widths")
        ax[1].semilogy(fr, P, lw=.8, label=f"{x:.1f}{name}")
ax[1].set_xlabel("Hz"); ax[1].legend(ncol=5, fontsize=7); ax[1].set_title("Welch PSD per target (after 1 s settle)")
z = (t > 1.5) & (t < 5.5)
ax[2].plot(t[z], xr[z] - np.nanmean(xr[z]), ".-", lw=.8, ms=3, label="R"); ax[2].plot(t[z], xl[z] - np.nanmean(xl[z]), ".-", lw=.8, ms=3, label="L")
ax[2].set_title("zoom: centre target, 4 s"); ax[2].legend()
plt.tight_layout(); out = f.with_suffix(".png"); plt.savefig(out, dpi=90); print("plot", out)
# gain: how far does the mean move per target step (to compare against oscillation size)
means = {}
for x, n, mu, *_ in segs: means.setdefault(n, []).append((x, mu))
for n, v in means.items():
    v = sorted(v); xs, mus = np.array(v).T
    print(f"eye {n}: slope {np.polyfit(xs, mus, 1)[0]:.3f} eye-widths per screen-width")
