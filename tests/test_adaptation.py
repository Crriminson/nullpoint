"""Few-shot adaptation of the glance model and the board's 10-second setup / self-correction."""
import numpy as np

from nullpoint.models.glance import GlanceModel
from nullpoint.board.selector import Selector, SelectorConfig, SETUP_ORDER, SETUP_HOLD_S

D = 17
ZX = {0: -1.0, 1: 0.0, 2: 1.0}       # zone -> mean of the "iris x" features


def session(n=200, shift=0.0, seed=0):
    """Toy gaze features: columns 0 and 3 carry the zone, everything shifts by `shift` (head moved)."""
    r = np.random.default_rng(seed)
    y = r.integers(0, 3, n)
    G = r.normal(0, 0.15, (n, D)).astype(np.float32)
    G[:, 0] += [ZX[k] for k in y]; G[:, 3] += [ZX[k] for k in y]
    G[:, :6] += shift
    return G, y


def test_recalibrate_leaves_a_model_that_still_fits_alone():
    gm = GlanceModel("lda").fit(*session())
    G, y = session(shift=0.05, seed=3)
    info = gm.recalibrate(G, y)
    assert info["ok"] and not info["changed"] and np.allclose(gm.b, 0) and np.allclose(gm.a, 1)


def test_recalibrate_undoes_a_session_shift():
    gm = GlanceModel("lda").fit(*session())
    Gb, yb = session(shift=0.6, seed=1)
    before = (gm.proba(Gb).argmax(1) == yb).mean()
    few = np.concatenate([np.flatnonzero(yb == k)[:30] for k in range(3)])   # ~1 s per zone
    info = gm.recalibrate(Gb[few], yb[few])
    after = (gm.proba(Gb).argmax(1) == yb).mean()
    assert info["ok"] and before < 0.8 and after > 0.95


def test_nudge_corrects_from_centre_only_and_skips_moving_eyes():
    gm = GlanceModel("lda").fit(*session())
    Gb, yb = session(shift=0.6, seed=2)
    still = Gb[yb == 1][:20]
    n = 0
    while gm.nudge(still, 1) is not None and n < 20:   # stops by itself once the centre looks like centre
        n += 1
    assert 1 <= n < 20
    assert gm.proba(still)[:, 1].mean() >= 0.8
    assert (gm.proba(Gb).argmax(1) == yb).mean() > 0.95
    b = gm.b.copy()
    moving = np.vstack([Gb[yb == 2][:10], Gb[yb == 1][:10]])     # a glance coming back from the right
    assert gm.nudge(moving, 1) is None and np.allclose(gm.b, b)


class _Blink:
    """Scripted blink model: returns `kind` at the given times."""
    def __init__(self, script): self.script = dict(script)
    def step(self, t, b):
        for ts, kind in list(self.script.items()):
            if t >= ts:
                del self.script[ts]; return kind, {"p": 1.0}
        return None, {}


def _run(sel, Gfn, t_end, fs=30):
    for i in range(int(t_end * fs)):
        t = i / fs
        sel.update(t, Gfn(t), np.zeros(8, np.float32))


def test_setup_runs_aligns_and_reports():
    gm = GlanceModel("lda").fit(*session())
    infos = []
    sel = Selector(gm, _Blink({1.0: "deliberate"}), SelectorConfig(adapt=False), on_setup=infos.append)
    sel.offer_setup()
    assert sel.state.setup == {"phase": "ready"}
    order = SETUP_ORDER

    def G(t):        # the user looks where the dot is, from a shifted head position
        el = t - 1.0
        z = order[min(int(el // SETUP_HOLD_S), len(order) - 1)] if el >= 0 else "C"
        g = np.full(D, 0.6, np.float32); g[6:] = 0
        g[0] += ZX["LCR".index(z)]; g[3] += ZX["LCR".index(z)]
        return g
    _run(sel, G, 1.0 + SETUP_HOLD_S * len(order) + 0.5)
    assert infos and infos[0]["ok"] and sel.state.setup_ok
    assert sel.state.setup["phase"] == "done"
    assert not [e for e in sel.events if e[1] == "step"]        # nothing moved during the setup


def test_emergency_cancels_setup():
    gm = GlanceModel("lda").fit(*session())
    sel = Selector(gm, _Blink({0.5: "deliberate", 2.0: "emergency"}), SelectorConfig(adapt=False))
    sel.offer_setup()
    _run(sel, lambda t: np.zeros(D, np.float32), 3.0)
    assert sel.state.emergency and sel.state.setup is None


def test_new_user_setup_fits_a_model_from_scratch():
    gm = GlanceModel("lda", cols=[0, 1, 2, 3, 4, 5])          # never fitted: a brand-new user
    infos = []
    sel = Selector(gm, _Blink({0.5: "deliberate"}), SelectorConfig(adapt=False), on_setup=infos.append)
    sel.offer_setup()
    rng = np.random.default_rng(4)

    def G(t):
        el = t - 0.5
        z = SETUP_ORDER[min(int(el // SETUP_HOLD_S), len(SETUP_ORDER) - 1)] if el >= 0 else "C"
        g = rng.normal(0, 0.1, D).astype(np.float32)
        g[0] += ZX["LCR".index(z)]; g[3] += ZX["LCR".index(z)]
        return g
    _run(sel, G, 0.5 + SETUP_HOLD_S * len(SETUP_ORDER) + 0.3)
    assert infos and infos[0]["fitted"] and gm.fitted and gm.adaptable
    Gt, yt = session(seed=5)
    Gt[:, [1, 2, 4, 5]] = rng.normal(0, 0.1, (len(Gt), 4))
    assert (gm.proba(Gt).argmax(1) == yt).mean() > 0.95
