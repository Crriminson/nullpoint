"""Simulator, live Adapt switch, repeat cap, setup re-learning and the phrase test."""
import numpy as np

from nullpoint.models.glance import GlanceModel
from nullpoint.board.selector import Selector, SelectorConfig, MAX_REPEAT, SETUP_ORDER, SETUP_HOLD_S
from nullpoint.board.task import TaskRunner, TARGETS, BLOCK_PAUSE_S, GAP_S, summarize
from nullpoint.board.phrases import PAGES
from nullpoint.config import screen_x_to_deg

D = 17
ZONE_X = {"L": 0.17, "C": 0.5, "R": 0.83}
GAIN = 0.006                                  # feature units per degree, like the real iris features


def feat(deg, rng=None, sd=0.0):
    g = np.zeros(D, np.float32)
    g[0] = g[3] = 0.5 + GAIN * deg + (rng.normal(0, sd) if rng is not None else 0)
    return g


def model(rng=np.random.default_rng(0), sign=1):
    X, y = [], []
    for k, z in enumerate("LCR"):
        for _ in range(60):
            X.append(feat(sign * screen_x_to_deg(ZONE_X[z]), rng, 0.01)); y.append(k)
    gm = GlanceModel("lda", cols=[0, 1, 2, 3, 4, 5]).fit(np.array(X), np.array(y))
    gm.zone_x = dict(ZONE_X)
    return gm


class NoBlink:
    def step(self, t, b): return None, {}


def run(sel, deg_fn, t0, t1, fs=30, sd=0.005):
    rng = np.random.default_rng(1)                         # landmark noise, as in real frames
    for i in range(int(t0 * fs), int(t1 * fs)):
        t = i / fs
        sel.update(t, feat(deg_fn(t), rng, sd), np.zeros(8, np.float32))


def test_simulator_shakes_ordinary_smoothing_more_than_nullpoint():
    sel = Selector(model(), NoBlink(), SelectorConfig(adapt=True))
    sel.set_sim(15.0)
    xs = {"plain": [], "adapt": []}
    for i in range(30 * 8):
        sel.update(i / 30, feat(0.0), np.zeros(8, np.float32))
        if i > 60:
            for k in xs:
                xs[k].append(sel.state.strip[k])
    span = {k: np.ptp(np.percentile(v, [5, 95])) for k, v in xs.items()}
    assert span["plain"] > 2 * span["adapt"] and span["plain"] > 0.1     # 15 deg shake vs steadied
    assert sel.state.sim["amp"] == 15.0 and sel._freq() == 3.0


def test_adapt_switches_live():
    sel = Selector(model(), NoBlink(), SelectorConfig(adapt=True, freq_hz=None))
    sel.set_sim(10.0)
    assert sel.state.adapt and abs(sel.dwell - 1 / 3) < 1e-9          # one 3 Hz period
    sel.set_adapt(False)
    assert not sel.state.adapt and sel.dwell == sel.cfg.dwell_s


def test_looking_away_steps_at_most_max_repeat_times():
    sel = Selector(model(), NoBlink(), SelectorConfig(adapt=False))
    run(sel, lambda t: screen_x_to_deg(0.9), 0, 12)                      # stares right for 12 s
    assert sum(1 for e in sel.events if e[1] == "step") == MAX_REPEAT


def test_setup_relearns_when_aligning_is_not_enough():
    gm = model(sign=-1)                                                    # trained on a mirrored image
    infos = []

    class Blink:
        done = False
        def step(self, t, b):
            if t >= 0.5 and not self.done:
                self.done = True; return "deliberate", {"p": 1.0}
            return None, {}
    sel = Selector(gm, Blink(), SelectorConfig(adapt=False, setup_x=ZONE_X), on_setup=infos.append)
    sel.offer_setup()

    def gaze(t):
        el = t - 0.5
        z = SETUP_ORDER[min(int(el // SETUP_HOLD_S), len(SETUP_ORDER) - 1)] if el >= 0 else "C"
        return screen_x_to_deg(ZONE_X[z])
    run(sel, gaze, 0, 0.5 + SETUP_HOLD_S * len(SETUP_ORDER) + 0.3)
    assert infos and infos[0].get("refit") and infos[0]["train_acc"] > 0.95


class FakeSel:
    """Just enough of the Selector for the task runner."""
    class S:
        page, cursor, task = 0, 5, None
    class G:
        class H:
            def reset(self): pass
        hmm = H()
    def __init__(self):
        self.state, self.gm, self.adapt_calls, self.saved = self.S(), self.G(), [], None
    def set_adapt(self, on): self.adapt_calls.append(on)
    def on_task(self, summary, trials):
        self.saved = (summary, trials); return "reports/task-test.json"


def test_phrase_test_runs_abba_and_summarises():
    sel, tr = FakeSel(), TaskRunner(sim_amp=10.0, adapt_before=True, seed=1)
    tiles = PAGES[0][1]
    t = 0.0
    tr.start(t, sel)
    assert tr.order in (["off", "on", "on", "off"], ["on", "off", "off", "on"])
    n_done = 0
    while not tr.finished and t < 600:
        t += 0.1
        tr.tick(t, sel)
        if tr.phase == "trial" and t - tr.cur["t0"] > 1.0:
            assert sel.state.page == 0 and tr.cur["target"] in TARGETS
            if tr.cur["adapt"] == "off" and tr.cur["wrong"] == 0:
                tr.on_select(t, tiles[0], "eyes", sel)                   # one wrong pick when off
            else:
                for _ in range(tr.cur["min_steps"]):
                    tr.on_step(t, 1, "eyes")
                tr.on_select(t, tr.cur["target_text"], "eyes", sel)
                n_done += 1
    summary, trials = sel.saved
    assert tr.finished and n_done == len(trials) == 4 * len(TARGETS)
    assert summary["off"]["wrong_selections_per_phrase"] == 1.0 and summary["on"]["wrong_selections_per_phrase"] == 0.0
    assert summary["meta"]["sim_amp_deg"] == 10.0 and not summary["meta"]["keyboard_used"]
    assert sel.adapt_calls[:4] == [c == "on" for c in tr.order] and sel.adapt_calls[-1] is True
    assert summarize(trials)["on"]["found"] == 2 * len(TARGETS)
