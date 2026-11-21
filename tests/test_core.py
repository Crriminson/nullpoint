"""Fast unit tests for the logic that does not need a camera."""
import numpy as np
import pytest

from nullpoint.scope import synth, metrics
from nullpoint.models.blink import detect_events, BlinkModel, DurationThreshold, EMERGENCY_S, EMERGENCY_MAX_S
from nullpoint.adapt.filters import PeriodMean, FoveationMean, apply
from nullpoint.board.selector import Selector, SelectorConfig
from nullpoint.board.phrases import PAGES, NEXT
from nullpoint.config import deg_to_screen_x, screen_x_to_deg


@pytest.mark.parametrize("kind,f,amp", [("pendular", 3.0, 4.0), ("jerk_r", 2.5, 6.0), ("jerk_l", 4.0, 3.0)])
def test_metrics_recover_frequency_and_amplitude(kind, f, amp):
    _, x, _ = synth.generate(synth.NystParams(kind=kind, freq_hz=f, amp_deg=amp, jitter=0.03), 6, 240, seed=1, noise_deg=0.1)
    m = metrics.analyse(x, 240)
    assert abs(m.freq_hz - f) / f < 0.1
    assert abs(m.amp_deg - amp) / amp < 0.25


def test_jerk_direction():
    _, x, _ = synth.generate(synth.NystParams(kind="jerk_l", freq_hz=3, amp_deg=6), 6, 240, seed=2, noise_deg=0.1)
    assert metrics.analyse(x, 240).fast_dir == "left"


def test_foveation_gaze_beats_mean_for_jerk():
    _, x, _ = synth.generate(synth.NystParams(kind="jerk_r", freq_hz=3, amp_deg=8), 6, 240, seed=3, noise_deg=0.1)
    m = metrics.analyse(x, 240)
    assert abs(m.gaze_foveation_deg) < abs(m.gaze_mean_deg)


def _blinks(fs=30, spec=((1, .15), (3, .2), (6, .7))):
    t = np.arange(0, 10, 1 / fs); B = np.zeros((len(t), 8), np.float32); B[:, 4:6] = .3
    for a, d in spec:
        m = (t >= a) & (t < a + d); B[m, 0] = B[m, 1] = .9; B[m, 4:6] = .02
    return t, B


def test_blink_events_and_threshold_model():
    t, B = _blinks()
    ev = detect_events(t, B)
    assert len(ev) == 3
    X = np.array([e.feats for e in ev]); clf = DurationThreshold().fit(X, np.array([0, 0, 1]))
    out = [BlinkModel(clf, "thr").step(tt, b) for tt, b in zip(t, B)]
    bm = BlinkModel(clf, "thr"); kinds = [k for k, _ in (bm.step(tt, b) for tt, b in zip(t, B)) if k]
    assert kinds == ["natural", "natural", "deliberate"]


def _thr_model():
    return BlinkModel(DurationThreshold().fit(np.array([[0.1] + [0] * 12, [0.8] + [0] * 12]), np.array([0, 1])), "thr")


def test_emergency_fires_on_reopening_after_a_long_closure():
    t, B = _blinks(spec=((2, EMERGENCY_S + 0.8),))
    bm = _thr_model()
    kinds = [k for k, _ in (bm.step(tt, b) for tt, b in zip(t, B)) if k]
    assert kinds == ["emergency_pending", "emergency"]   # cue at 2 s, alarm on opening, no selection


def test_dozing_off_never_raises_the_alarm():
    fs = 30
    t = np.arange(0, 20, 1 / fs); B = np.zeros((len(t), 8), np.float32); B[:, 4:6] = .3
    m = (t >= 2) & (t < 2 + EMERGENCY_MAX_S + 5); B[m, 0] = B[m, 1] = .9; B[m, 4:6] = .02
    bm = _thr_model()
    kinds = [k for k, _ in (bm.step(tt, b) for tt, b in zip(t, B)) if k]
    assert kinds == ["emergency_pending", "rest"]


def test_period_mean_cancels_periodic_component():
    fs, f = 30.0, 3.0
    t = np.arange(0, 5, 1 / fs); x = 5 + 2 * np.sin(2 * np.pi * f * t)
    y = apply(PeriodMean(fs, f), x)[60:]
    assert np.abs(y - 5).max() < 0.05


def test_foveation_mean_reduces_jerk_bias():
    fs = 30.0
    _, x, _ = synth.generate(synth.NystParams(kind="jerk_r", freq_hz=3, amp_deg=6), 8, fs, seed=4)
    plain = apply(PeriodMean(fs, 3.0), x)[60:]; fov = apply(FoveationMean(fs, 1 / 3), x)[60:]
    assert abs(np.median(fov)) < abs(np.median(plain))


def test_selector_keyboard_flow():
    class G:
        class H:
            def step(self, p): return np.array(p)
        hmm = H()
        def proba(self, g): return np.array([[0, 1, 0]])
    class Bm:
        def step(self, t, b): return None, {}
    said = []
    s = Selector(G(), Bm(), SelectorConfig(adapt=False), speak=said.append)
    s._step(1, 0.0); s.select(0.1)
    assert said == [PAGES[0][1][1]]
    s.state.cursor = PAGES[0][1].index(NEXT); s.select(1.0)
    assert s.state.page == 1 and s.state.cursor == 0


def test_screen_geometry_roundtrip():
    for d in (-15, -5, 0, 7.5, 15):
        assert abs(screen_x_to_deg(deg_to_screen_x(d)) - d) < 1e-9


def test_gaze_stabilizer_dropin():
    from nullpoint.adapt import GazeStabilizer
    fs = 30.0
    _, x, _ = synth.generate(synth.NystParams(kind="pendular", freq_hz=3.0, amp_deg=8.0), 6, fs, seed=1)
    stab = GazeStabilizer.from_profile({"freq_hz": 3.0, "null_deg": 5.0}, fs=fs)
    y = np.array([stab(v) for v in x])[60:]
    assert np.std(y) < 0.25 * np.std(x[60:])
    assert abs(stab.dwell_s - 1 / 3) < 1e-9 and stab.null_screen_x > 0.5
    assert GazeStabilizer.from_profile({}).dwell_s == 0.4                       # no profile: defaults
