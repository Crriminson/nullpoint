"""Replay benchmark: recorded webcam gaze streams + injected synthetic nystagmus.

Question: as nystagmus gets stronger, does the board still move the highlight the right way,
and does the profile-driven Adapt chain help?

Data: two webcam grid scans of a test user (3x5 dot grid, 4 s per dot). Middle-row dots at screen
x = 0.1 / 0.5 / 0.9 become zones L / C / R. The glance model (LDA + HMM) is trained on scan A
(clean) and the selector is run on scan A itself (default: isolates the effect of nystagmus) or on
scan B (--cross: adds real session-to-session drift) with nystagmus injected into the iris-position
features. Injection is in degrees, converted to feature units with the per-feature gain
measured on scan A (feature change per degree of gaze, from the known dot angles).

Per condition (waveform x amplitude) and Adapt off / on we report:
  * hold accuracy  : fraction of L/R holds that produce >= 1 step in the correct direction
  * wrong steps/min: steps in the wrong direction, or any step during a centre hold
  * latency        : time from hold start (+ 0.3 s reaction grace) to the first correct step
Steps in the first 0.3 s after a dot moves are not scored: human saccade latency is ~200 ms, so
they belong to the previous hold's intent (found when a lagged filter's auto-repeat landed 40 ms
after the dot moved and was counted as an error).
Writes reports/replay_<same|cross>[_eye].json/.png (--eye: glance model on eye-in-head features only).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nullpoint.config import RAW, REPORTS, screen_x_to_deg
from nullpoint.capture import FaceFrame
from nullpoint.features import gaze_features
from nullpoint.models.glance import GlanceModel, FEATURE_SETS
from nullpoint.board.selector import Selector, SelectorConfig
from nullpoint.scope import synth

LAYOUTS = {"wide": {0.1: 0, 0.5: 1, 0.9: 2}, "narrow": {0.3: 0, 0.5: 1, 0.7: 2}}
ZONE_X = LAYOUTS["wide"]


REACTION_S = 0.3   # grace after a dot moves: a step this early is the previous intent, not an error


class NoBlink:
    def step(self, t, b): return None, {}


def load_scan(path: Path):
    d = np.load(path)
    W, H = (int(v) for v in d["frame_wh"])
    t, tx, ty = d["t"], d["target"], d["target_y"]
    lms = d["lms"]          # read once: indexing an NpzFile re-decompresses the whole array each time
    G = np.full((len(t), 17), np.nan, np.float32)
    for i in range(len(t)):
        if np.isfinite(lms[i, 0, 0]):
            G[i] = gaze_features(FaceFrame(float(t[i]), lms[i], {}, None, (W, H), 0.0))
    row = np.isclose(ty, 0.5)
    return t, np.round(tx, 1), row, G


def train_frames(t, tx, row, G, settle=0.8, zx=None):
    zx = zx or ZONE_X
    keep = np.zeros(len(t), bool)
    for x in zx:
        idx = np.flatnonzero(row & (tx == x))
        if len(idx):
            keep[idx[t[idx] > t[idx[0]] + settle]] = True
    keep &= np.isfinite(G).all(1)
    return G[keep], np.array([zx[x] for x in tx[keep]])


def gains(t, tx, row, G):
    """Feature units per degree of horizontal gaze for each feature (slope over the 5 middle-row dots)."""
    xs, mus = [], []
    for x in (0.1, 0.3, 0.5, 0.7, 0.9):
        idx = np.flatnonzero(row & (tx == x)); idx = idx[t[idx] > t[idx[0]] + 0.8]
        xs.append(screen_x_to_deg(x)); mus.append(np.nanmedian(G[idx], 0))
    return np.polyfit(np.array(xs), np.array(mus), 1)[0]          # (17,)


def run_raw(t, tx, row, G, gm, gain, kind, amp, adapt, freq=3.0, seed=0, zx=None, score=None):
    """Run the selector over a scan with injected nystagmus. `score` maps hold x -> seconds after the
    dot moved from which steps count (default: every hold in zx, from REACTION_S). Returns counts."""
    zx = zx or ZONE_X
    score = score if score is not None else {x: REACTION_S for x in zx}
    fs = 1 / np.median(np.diff(t))
    _, nyst, _ = synth.generate(synth.NystParams(kind=kind, freq_hz=freq, amp_deg=amp), t[-1] + 1, 240, seed=seed)
    tt = np.arange(len(nyst)) / 240
    inj = np.interp(t, tt, nyst) if amp > 0 and kind != "none" else np.zeros(len(t))
    Gi = G.copy()
    for j in (0, 3):                       # iris x of both eyes (eye-in-head horizontal)
        Gi[:, j] = G[:, j] + gain[j] * inj
    gm.hmm.reset()
    sel = Selector(gm, NoBlink(), SelectorConfig(adapt=adapt, freq_hz=freq if adapt else None, fs=fs))
    for i in range(len(t)):
        g = Gi[i] if np.isfinite(Gi[i]).all() else None
        sel.update(float(t[i]), g, None)
    steps = [(e[0], e[2]) for e in sel.events if e[1] == "step"]
    # evaluate per middle-row hold
    ok, lat, wrong, dur_min = [], [], 0, 0.0
    for x, off in score.items():
        z = zx[x]
        idx = np.flatnonzero(row & (tx == x))
        if not len(idx):
            continue
        t0, t1 = t[idx[0]] + off, t[idx[-1]]     # eyes cannot respond before ~200-300 ms
        dur_min += (t1 - t0) / 60
        s_in = [d for (ts, d) in steps if t0 <= ts <= t1]
        want = {0: -1, 1: 0, 2: 1}[z]
        if want == 0:
            wrong += len(s_in)
        else:
            good = [ts for (ts, d) in steps if t0 <= ts <= t1 and d == want]
            wrong += sum(1 for d in s_in if d != want)
            ok.append(len(good) > 0)
            if good:
                lat.append(good[0] - t0)
    return {"ok": ok, "lat": lat, "wrong": wrong, "dur_min": dur_min}


def summarize(raws):
    """Pool counts from several runs into hold accuracy, wrong steps/min and latency."""
    ok = [o for r in raws for o in r["ok"]]
    lat = [v for r in raws for v in r["lat"]]
    dur = sum(r["dur_min"] for r in raws)
    return {"hold_acc": float(np.mean(ok)) if ok else np.nan,
            "wrong_per_min": sum(r["wrong"] for r in raws) / max(dur, 1e-9),
            "latency_s": float(np.mean(lat)) if lat else np.nan}


def run(*args, **kw):
    return summarize([run_raw(*args, **kw)])


SAME_SESSION = "--cross" not in sys.argv
EYE = "--eye" in sys.argv          # eye-in-head features only (the board default since eval/recal.py)


def main():
    scans = sorted(RAW.glob("scan-*-grid-*.npz"))
    if len(scans) < 2:
        raise SystemExit("need two grid scans in data/raw")
    A, B = (load_scan(p) for p in scans[:2])
    gain = gains(*A)
    print("feature gain per degree (iris x R/L):", np.round(gain[[0, 3]], 4))
    amps = [0, 2, 4, 6, 8, 12, 15]
    res = {}
    for lay, zx in LAYOUTS.items():
        Ga, ya = train_frames(*A, zx=zx)
        gm = GlanceModel("lda", cols=FEATURE_SETS["eye_geom"] if EYE else None).fit(Ga, ya)
        for kind in ("jerk_r", "pendular"):
            for adapt in (False, True):
                key = f"{lay}_{kind}_{'adapt' if adapt else 'plain'}"
                res[key] = []
                for amp in amps:
                    test = A if SAME_SESSION else B
                    rs = [run(*test, gm, gain, kind, amp, adapt, freq=f, seed=s, zx=zx) for s in range(2) for f in (2.0, 3.0, 4.0)]
                    m = {k: float(np.nanmean([r[k] for r in rs])) for k in rs[0]}
                    m["amp"] = amp; res[key].append(m)
                    print(f"{key:26s} amp {amp:4.1f}: hold acc {m['hold_acc']:.2f}  wrong/min {m['wrong_per_min']:5.2f}  latency {m['latency_s']:.2f}s")
    REPORTS.mkdir(exist_ok=True)
    tag = ("same" if SAME_SESSION else "cross") + ("_eye" if EYE else "")
    (REPORTS / f"replay_{tag}.json").write_text(json.dumps({"scans": ["scan_A.npz", "scan_B.npz"], "freqs_hz": [2, 3, 4], "results": res}, indent=1))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for r_, lay in enumerate(LAYOUTS):
        for kind, c in (("jerk_r", "#c0392b"), ("pendular", "#2471a3")):
            for adapt, ls in ((False, "--"), (True, "-")):
                key = f"{lay}_{kind}_{'adapt' if adapt else 'plain'}"
                a = [r["amp"] for r in res[key]]
                lab = f"{kind} {'Adapt on' if adapt else 'Adapt off'}"
                ax[r_, 0].plot(a, [r["hold_acc"] for r in res[key]], ls, c=c, marker="o", label=lab)
                ax[r_, 1].plot(a, [r["wrong_per_min"] for r in res[key]], ls, c=c, marker="o", label=lab)
        ax[r_, 0].set_title(f"{lay} layout: L/R holds that step correctly"); ax[r_, 0].set_ylim(0, 1.05)
        ax[r_, 1].set_title(f"{lay} layout: wrong steps per minute")
        for a_ in ax[r_]:
            a_.set_xlabel("injected nystagmus (deg peak-to-peak, 2-4 Hz)"); a_.grid(alpha=.3)
    ax[0, 0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(REPORTS / f"replay_{tag}.png", dpi=110)


if __name__ == "__main__":
    main()
