"""Webcam null-zone finder on synthetic posture streams with a known answer."""
import numpy as np

from nullpoint.adapt.nullzone import NullFinder, schedule, total_s
from nullpoint.config import screen_x_to_deg, zone_targets

DOTS = zone_targets(0.5)
EYE_G, HEAD_G = 0.008, 0.004          # feature units per degree (eye iris x, nose offset)


def run(head_turn_deg, fps=30, noise=0.002, seed=0, drop=0.0):
    """User whose habitual head turn while reading is `head_turn_deg` (eyes counter-rotate)."""
    rng = np.random.default_rng(seed)
    nf = NullFinder(0.0, DOTS)
    for i in range(int((total_s() + 0.5) * fps)):
        t = i / fps
        s = schedule(t)
        g = None
        if s is not None and rng.random() >= drop:
            ph, key, _, into = s
            if ph == "eyes":
                eye, head = screen_x_to_deg(DOTS[key]), 0.0
            elif ph == "head":
                eye, head = 0.0, screen_x_to_deg(DOTS[key])
            else:
                eye, head = -head_turn_deg, head_turn_deg
            g = np.zeros(17, np.float32)
            g[0] = g[3] = 0.45 + EYE_G * eye + rng.normal(0, noise)
            g[9] = 0.01 + HEAD_G * head + rng.normal(0, noise)
        nf.step(t, g)
    assert nf.done
    return nf.result


def test_finds_a_null_zone_from_head_posture():
    r = run(head_turn_deg=10.0)
    assert r["ok"] and r["clear"] and r["agree"]
    assert abs(r["null_deg"] - (-10.0)) < 1.0 and abs(r["head_check_deg"] - (-10.0)) < 1.5


def test_no_head_turn_means_no_clear_null():
    r = run(head_turn_deg=0.0)
    assert r["ok"] and not r["clear"] and abs(r["null_deg"]) < 1.0


def test_fails_cleanly_without_a_face():
    r = run(head_turn_deg=10.0, drop=1.0)
    assert not r["ok"]
