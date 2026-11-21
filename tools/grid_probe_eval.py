"""Grid-scan feasibility: how many gaze targets can webcam landmarks separate, and where is jitter lowest?

Train on one grid run, test on another (different session = honest estimate).
Usage: python grid_probe_eval.py TRAIN.npz TEST.npz
"""
import sys
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def features(path):
    d = np.load(path); L = d["lms"].copy(); W, H = d["frame_wh"]; L[..., 0] *= W; L[..., 1] *= H
    t, tx, ty = d["t"], d["target"], d["target_y"]
    feats = []
    for out, inn, iris, top, bot in ((33, 133, 468, 159, 145), (263, 362, 473, 386, 374)):
        a, b = L[:, out, :2], L[:, inn, :2]; w = np.linalg.norm(b - a, axis=1)
        u = (b - a) / w[:, None]; n = np.stack([-u[:, 1], u[:, 0]], 1)
        rel = L[:, iris, :2] - a
        feats += [np.einsum("ij,ij->i", rel, u) / w, np.einsum("ij,ij->i", rel, n) / w,
                  np.linalg.norm(L[:, top, :2] - L[:, bot, :2], axis=1) / w]   # lid opening tracks vertical gaze
    # head pose proxy: nose tip relative to eye-corner midpoint
    mid = L[:, [33, 133, 362, 263], :2].mean(1); iod = np.linalg.norm(L[:, 33, :2] - L[:, 263, :2], axis=1)
    feats += list(((L[:, 1, :2] - mid) / iod[:, None]).T)
    X = np.stack(feats, 1)
    keep = np.zeros(len(t), bool)
    key = np.round(tx * 10) * 100 + np.round(ty * 100)
    for k in np.unique(key):
        idx = np.where(key == k)[0]; keep[idx[t[idx] > t[idx[0]] + 1.0]] = True   # 1 s settle
    keep &= np.isfinite(X).all(1) & (d["blinkL"] < 0.4)
    return X[keep], tx[keep], ty[keep]

Xa, xa, ya = features(sys.argv[1]); Xb, xb, yb = features(sys.argv[2])
def run(name, la, lb, ma=None, mb=None):
    ma = np.ones(len(la), bool) if ma is None else ma; mb = np.ones(len(lb), bool) if mb is None else mb
    for clf in (LinearDiscriminantAnalysis(), RandomForestClassifier(300, min_samples_leaf=5, random_state=0)):
        la = np.round(la * 100).astype(int); lb = np.round(lb * 100).astype(int)
        clf.fit(Xa[ma], la[ma]); p = clf.predict(Xb[mb])
        # per-frame accuracy and majority-vote over 0.5 s windows (what a selector would actually use)
        lab = lb[mb]; win = 11; votes = []
        for i in range(0, len(p) - win, win // 2):
            seg = p[i:i + win]
            if len(set(lab[i:i + win])) == 1:
                v, c = np.unique(seg, return_counts=True); votes.append(v[c.argmax()] == lab[i])
        print(f"{name:34s} {type(clf).__name__[:12]:12s} frame acc {accuracy_score(lab, p):.2f}   0.5 s vote acc {np.mean(votes):.2f}")
mid_row_a, mid_row_b = ya == 0.5, yb == 0.5
run("5 columns (middle row)", np.round(xa, 1), np.round(xb, 1), mid_row_a, mid_row_b)
c3a, c3b = np.isin(np.round(xa, 1), [0.1, 0.5, 0.9]) & mid_row_a, np.isin(np.round(xb, 1), [0.1, 0.5, 0.9]) & mid_row_b
run("3 columns L/C/R (middle row)", np.round(xa, 1), np.round(xb, 1), c3a, c3b)
cc_a, cc_b = np.round(xa, 1) == 0.5, np.round(xb, 1) == 0.5
run("3 rows up/mid/down (centre column)", ya, yb, cc_a, cc_b)
run("all 15 grid cells", np.round(xa, 1) * 10 + ya, np.round(xb, 1) * 10 + yb)

print("\njitter per target (std of (R-L)/2 horizontal iris feature, after settle), run B:")
gain = np.polyfit(xb[mid_row_b], (Xb[mid_row_b, 0] - Xb[mid_row_b, 3]) / 2, 1)[0]
for y in (0.15, 0.5, 0.85):
    row = []
    for x in (0.1, 0.3, 0.5, 0.7, 0.9):
        m = (np.round(xb, 1) == x) & (yb == y)
        s = (Xb[m, 0] - Xb[m, 3]) / 2
        row.append(f"{np.std(s) / abs(gain):.3f}")
    print(f"  row y={y:.2f}: " + "  ".join(row) + "   (in screen-widths of equivalent gaze)")
