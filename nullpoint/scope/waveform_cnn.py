"""1D-CNN nystagmus waveform classifier (none / pendular / jerk_l / jerk_r).

Trained entirely on synthetic traces (scope.synth), where labels are exact, then applied to real
traces. Input: a 2 s window of eye position (deg), resampled to the model's rate, de-meaned.
Position and its first difference (velocity) are the two input channels, because fast phases
are defined by velocity.

    python -m nullpoint.scope.waveform_cnn --fs 120     # train + evaluate vs heuristic
    python -m nullpoint.scope.waveform_cnn --fs 30
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch import nn

from nullpoint.config import ROOT, REPORTS
from nullpoint.scope import synth, metrics

WIN_S = 2.0
MODEL_DIR = ROOT / "nullpoint" / "scope" / "weights"   # small, synthetic-only -> safe to commit


class WaveNet(nn.Module):
    def __init__(self, n_cls=4):
        super().__init__()
        def blk(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
        self.f = nn.Sequential(blk(2, 24, 7), blk(24, 48, 5), blk(48, 64, 5), blk(64, 64, 3))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.head(self.f(x))


def prep(X: np.ndarray, fs: float) -> torch.Tensor:
    X = X - X.mean(1, keepdims=True)
    V = np.diff(X, axis=1, prepend=X[:, :1]) * fs / 100.0      # velocity, scaled to ~unit range
    s = np.clip(X.std(1, keepdims=True), 0.3, None)              # per-window position scaling
    return torch.tensor(np.stack([X / s, V / np.clip(np.abs(V).std(1, keepdims=True), 0.3, None)], 1), dtype=torch.float32)


def train(fs: float, n_train=24000, n_val=3000, epochs=12, seed=0, noise=(0.05, 1.0)):
    torch.manual_seed(seed)
    Xtr, ytr, _ = synth.dataset(n_train, fs, WIN_S, seed=seed, noise_range=noise)
    Xva, yva, pva = synth.dataset(n_val, fs, WIN_S, seed=seed + 1, noise_range=noise)
    tr, va = prep(Xtr, fs), prep(Xva, fs)
    ytr_t, yva_t = torch.tensor(ytr), torch.tensor(yva)
    m = WaveNet(); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=epochs * (len(tr) // 128 + 1))
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(tr)); t0 = time.time()
        for i in range(0, len(tr), 128):
            b = perm[i:i + 128]
            opt.zero_grad(); l = lossf(m(tr[b]), ytr_t[b]); l.backward(); opt.step(); sched.step()
        m.eval()
        with torch.no_grad():
            acc = (m(va).argmax(1) == yva_t).float().mean().item()
        print(f"  epoch {ep + 1:2d}  loss {l.item():.3f}  val acc {acc:.3f}  ({time.time() - t0:.1f}s)")
    return m, (Xva, yva, pva)


def evaluate(m, Xva, yva, fs):
    m.eval()
    with torch.no_grad():
        pred = m(prep(Xva, fs)).argmax(1).numpy()
    heur = np.array([synth.CLASSES.index(metrics.analyse(x, fs).waveform_heuristic) for x in Xva])
    cm = np.zeros((4, 4), int)
    for t, p in zip(yva, pred):
        cm[t, p] += 1
    return {"cnn_acc": float((pred == yva).mean()), "heuristic_acc": float((heur == yva).mean()),
            "cnn_confusion": cm.tolist(), "classes": synth.CLASSES}


def load(fs: float) -> WaveNet:
    m = WaveNet(); m.load_state_dict(torch.load(MODEL_DIR / f"wavenet_{int(fs)}hz.pt", weights_only=True)); m.eval()
    return m


def classify(x: np.ndarray, fs_in: float, fs_model: float = 120) -> tuple[str, np.ndarray]:
    """Classify a real trace (any length >= 2 s): majority over 2 s windows (1 s hop)."""
    from scipy.signal import resample_poly
    from fractions import Fraction
    fr = Fraction(fs_model / fs_in).limit_denominator(50)
    y = resample_poly(np.asarray(x, float), fr.numerator, fr.denominator)
    W = int(WIN_S * fs_model)
    wins = np.array([y[i:i + W] for i in range(0, len(y) - W + 1, int(fs_model))])
    if not len(wins):
        return "none", np.zeros(4)
    with torch.no_grad():
        p = torch.softmax(load(fs_model)(prep(wins, fs_model)), 1).numpy().mean(0)
    return synth.CLASSES[int(p.argmax())], p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", type=float, default=120)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    noise = (0.05, 1.0) if a.fs >= 100 else (0.2, 1.2)
    m, (Xva, yva, _) = train(a.fs, epochs=a.epochs, noise=noise)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), MODEL_DIR / f"wavenet_{int(a.fs)}hz.pt")
    res = evaluate(m, Xva, yva, a.fs)
    print(json.dumps({k: v for k, v in res.items()}, indent=1))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"waveform_cnn_{int(a.fs)}hz.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
