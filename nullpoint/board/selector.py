"""Selection state machine: per-frame gaze/blink features -> board actions.

Interaction (designed for horizontal-only, nystagmus-tolerant control):
  * look LEFT / RIGHT (beyond the focus zone) and hold -> highlight steps one tile; keep holding
    to auto-repeat (at most MAX_REPEAT steps per glance, so looking away at someone in the room
    doesn't run the cursor round the page); look back at the focus card to stop
  * deliberate blink -> speak the highlighted phrase ("More options" changes page)
  * close the eyes for 2-8 s, then open them -> emergency. A soft cue sounds at 2 s; longer
    closures are rest or sleep and never raise the alarm (models/blink.py)
Glance updates are frozen while the eyelids are closing/closed: closure drags the iris
landmarks downward/sideways and would otherwise read as a glance.

Adapt (nystagmus profile) changes three things, all switchable for the off/on evaluation:
  1. feature filtering: velocity-weighted (foveation) average of the eye features over one period
  2. dwell time: a whole number of nystagmus periods (>= 0.3 s) instead of a fixed 0.4 s
  3. layout: the focus card (and the calibrated centre zone) sits at the null-point angle
Both filter banks (ordinary smoothing and Adapt) run on every frame; `cfg.adapt` picks which one
drives the board, so it can be switched live and the strip can always show both.

Staying calibrated without retraining (GlanceModel alignment):
  * 10-second setup: look at four dots (centre, left, right, centre; 2.5 s each, first 0.8 s
    ignored). With no model yet (a new user) it fits one from these ~7 s of frames; otherwise it
    checks the trained model, aligns it only if it misreads the dots, and re-learns from the dots
    if aligning still leaves it wrong (glasses on/off, very different light)
  * self-correction: just before a deliberate blink the user was reading the focus card, so those
    frames are a free "centre" label; each selection nudges the alignment a little (skipped if the
    eyes were still moving)

Demo and evaluation tools:
  * simulator: adds synthetic nystagmus (jerk, 3 Hz, 5/10/15 deg peak-to-peak) to the live iris
    features, converted to feature units with the glance model's own zone means (the replay
    benchmark's injection, live). Adapt is then tuned to 3 Hz, as a Scope profile would tune it.
  * strip: where ordinary smoothing and Adapt put the gaze on screen, for the explain view
  * task: the phrase-selection test for the end-to-end evaluation (board/task.py)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from nullpoint.adapt.filters import AdaptChain, PeriodMean
from nullpoint.adapt.nullzone import NullFinder
from nullpoint.config import zone_targets, screen_x_to_deg, deg_to_screen_x
from nullpoint.board.phrases import PAGES, NEXT, EMERGENCY_TEXT
from nullpoint.models.blink import ON as BLINK_ON
from nullpoint.scope import synth

EYE_DIMS = [0, 1, 2, 3, 4, 5]      # gaze feature columns that carry eye (not head) information
SETUP_ORDER = ["C", "L", "R", "C"]           # ends on the focus card, so nothing steps when it finishes
SETUP_HOLD_S, SETUP_SETTLE_S = 2.5, 0.8
CORRECT_WIN = (0.8, 0.15)          # self-correction window: from 0.8 s to 0.15 s before the eyelids closed
MAX_REPEAT = 4                     # steps per glance before the eyes must come back to the card
REFIT_BELOW = 0.9                  # setup: re-learn if aligning leaves < 90 % of the dot frames right
SIM_AMPS = [0.0, 5.0, 10.0, 15.0]  # simulator settings, degrees peak-to-peak
SIM_KIND, SIM_FREQ, SIM_LEN_S = "jerk_r", 3.0, 120.0


@dataclass
class SelectorConfig:
    adapt: bool = True
    freq_hz: float | None = None       # nystagmus frequency from the profile
    fs: float = 30.0
    belief_on: float = 0.8
    dwell_s: float = 0.4               # used when adapt is off / no profile
    repeat_s: float = 1.0
    refractory_s: float = 0.6          # after a selection, ignore glances briefly
    self_correct: bool = True
    setup_x: dict = field(default_factory=lambda: {"L": 0.17, "C": 0.5, "R": 0.83})


@dataclass
class BoardState:
    page: int = 0
    cursor: int = 0
    zone: str = "C"
    belief: list = field(default_factory=lambda: [0.0, 1.0, 0.0])
    last_spoken: str = ""
    emergency: bool = False
    alarm_pending: bool = False        # eyes closed > 2 s: opening them now calls for help
    history: list = field(default_factory=list)
    face: bool = False
    blink_p: float | None = None
    setup: dict | None = None          # 10-second setup: None | ready | run | done
    setup_ok: bool = False
    corrections: int = 0
    nullfind: dict | None = None       # webcam null-zone finder: None | ready | eyes/head/read | done
    adapt: bool = True                 # which filter drives the board right now
    sim: dict | None = None            # simulated nystagmus, if on
    strip: dict | None = None          # gaze on screen: ordinary smoothing vs Adapt
    task: dict | None = None           # phrase-selection test

    def tiles(self):
        return PAGES[self.page][1]

    def as_dict(self):
        return {"page": self.page, "page_name": PAGES[self.page][0], "tiles": self.tiles(), "cursor": self.cursor,
                "zone": self.zone, "belief": [round(float(b), 3) for b in self.belief], "last_spoken": self.last_spoken,
                "emergency": self.emergency, "alarm_pending": self.alarm_pending, "history": self.history[-12:],
                "face": self.face, "blink_p": self.blink_p, "setup": self.setup, "setup_ok": self.setup_ok,
                "corrections": self.corrections, "nullfind": self.nullfind, "adapt": self.adapt, "sim": self.sim,
                "strip": self.strip, "task": self.task}


class Selector:
    def __init__(self, glance_model, blink_model, cfg: SelectorConfig, speak=None, on_setup=None, on_nullzone=None,
                 on_task=None):
        self.gm, self.bm, self.cfg = glance_model, blink_model, cfg
        self.speak = speak or (lambda s: None)
        self.on_setup = on_setup or (lambda info: None)
        self.on_nullzone = on_nullzone or (lambda res: "")
        self.on_task = on_task or (lambda summary, trials: "")
        self.state = BoardState(adapt=cfg.adapt)
        self.sim_amp, self._sim = 0.0, None
        self._build_filters()
        self.hold_zone, self.hold_since, self.last_step, self.hold_steps = "C", 0.0, -1e9, 0
        self.block_until = 0.0
        self.events = []   # (t, kind, detail) for evaluation
        self.buf = deque(maxlen=90)        # recent (t, filtered features) for self-correction
        self._setup = None                 # running setup: {"t0", "X", "y"}
        self._null = None                  # running NullFinder
        self.task = None                   # TaskRunner (board/task.py)
        self.quiet = False                 # game page open: read the gaze, but don't step or speak

    # ------------------------------------------------------------------ filters, live Adapt switch
    def _freq(self):
        return SIM_FREQ if self.sim_amp > 0 else self.cfg.freq_hz

    def _build_filters(self):
        f, fs = self._freq(), self.cfg.fs
        self.f_adapt = [AdaptChain(fs, f, use_wflc=False) for _ in EYE_DIMS]
        self.f_plain = [PeriodMean(fs, None, default_s=0.1) for _ in EYE_DIMS]
        self.dwell = (max(1, int(np.ceil(0.3 * f))) / f) if (self.cfg.adapt and f) else self.cfg.dwell_s
        self.state.adapt = self.cfg.adapt

    def set_adapt(self, on: bool):
        if bool(on) != self.cfg.adapt:
            self.cfg.adapt = bool(on)
            self._build_filters()
            getattr(self.gm.hmm, "reset", lambda: None)()

    # ------------------------------------------------------------------ nystagmus simulator
    def set_sim(self, amp: float):
        self.sim_amp = float(amp)
        if self.sim_amp > 0:
            t, x, _ = synth.generate(synth.NystParams(kind=SIM_KIND, freq_hz=SIM_FREQ, amp_deg=self.sim_amp),
                                     SIM_LEN_S, 240, seed=7)
            self._sim = (t, x)
            self.state.sim = {"amp": self.sim_amp, "kind": "jerk", "freq": SIM_FREQ}
        else:
            self._sim, self.state.sim = None, None
        self._build_filters()

    def cycle_sim(self):
        i = SIM_AMPS.index(self.sim_amp) + 1 if self.sim_amp in SIM_AMPS else 0
        self.set_sim(SIM_AMPS[i % len(SIM_AMPS)])

    def _zone_x(self):
        return getattr(self.gm, "zone_x", None) or self.cfg.setup_x

    def _eye_lin(self):
        """Per eye: (model column, offset, slope) with aligned feature = offset + slope * degrees, from
        the model's zone means at the screen positions it was trained on. None without a fitted model."""
        mu, cols = getattr(self.gm, "mu", None), getattr(self.gm, "cols", None)
        if mu is None or getattr(self.gm, "a", None) is None:
            return None
        zx = self._zone_x()
        key = (mu.tobytes(), tuple(sorted(zx.items())))
        if getattr(self, "_lin_key", None) != key:
            deg = np.array([screen_x_to_deg(zx[z]) for z in "LCR"])
            try:
                lin = []
                for j in (0, 3):
                    p = cols.index(j) if cols is not None else j
                    c1, c0 = np.polyfit(deg, mu[:, p], 1)
                    lin.append((p, float(c0), float(c1)))
                self._lin = lin if all(abs(c1) > 1e-6 for _, _, c1 in lin) else None
            except ValueError:
                self._lin = None
            self._lin_key = key
        return self._lin

    def _inject(self, t, g):
        if self._sim is None or g is None:
            return g
        lin = self._eye_lin()
        if lin is None:
            return g
        st, sx = self._sim
        x = float(np.interp(t % SIM_LEN_S, st, sx))
        g = g.copy()
        for (p, _, c1), j in zip(lin, (0, 3)):
            g[j] += c1 * x / self.gm.a[p]          # degrees -> raw feature units
        return g

    def _strip(self, gp, ga):
        lin = self._eye_lin()
        if lin is None:
            return None

        def screen_x(gv):
            d = [((gv[j] * self.gm.a[p] + self.gm.b[p]) - c0) / c1 for (p, c0, c1), j in zip(lin, (0, 3))]
            return round(float(np.clip(deg_to_screen_x(float(np.mean(d))), 0.0, 1.0)), 4)
        zx = self._zone_x()
        return {"plain": screen_x(gp), "adapt": screen_x(ga),
                "bounds": [round((zx["L"] + zx["C"]) / 2, 4), round((zx["C"] + zx["R"]) / 2, 4)]}

    # ------------------------------------------------------------------ actions
    def _step(self, d, t, source="eyes"):
        n = len(self.state.tiles())
        self.state.cursor = (self.state.cursor + d) % n
        self.last_step = t
        self.events.append((t, "step", d))
        if self.task is not None and self.task.active:
            self.task.on_step(t, d, source)

    def select(self, t, source="eyes"):
        tile = self.state.tiles()[self.state.cursor]
        self.events.append((t, "select", tile))
        if self.state.emergency:
            self.state.emergency = False
            self.events.append((t, "emergency_cleared", ""))
            return
        if self.task is not None and self.task.active:
            self.task.on_select(t, tile, source, self)
            if tile == NEXT:                    # the test stays on the first page
                self.block_until = t + self.cfg.refractory_s
                return
        if tile == NEXT:
            self.state.page = (self.state.page + 1) % len(PAGES); self.state.cursor = 0
            return
        self.state.last_spoken = tile
        self.state.history.append(tile)
        self.speak(tile)
        self.block_until = t + self.cfg.refractory_s

    def emergency(self, t):
        if self._setup is not None:
            self.cancel_setup()               # safety first: an alarm always wins over the setup
        if self._null is not None:
            self.cancel_nullfind()
        self.state.emergency, self.state.alarm_pending = True, False
        self.events.append((t, "emergency", ""))
        self.speak(EMERGENCY_TEXT)

    # ------------------------------------------------------------------ staying calibrated
    @property
    def adaptable(self):
        return bool(getattr(self.gm, "adaptable", False))

    @property
    def can_setup(self):
        return hasattr(self.gm, "recalibrate") and hasattr(self.gm, "fit")

    @property
    def model_ready(self):
        return bool(getattr(self.gm, "fitted", True))

    def offer_setup(self):
        """Show the 'quick setup' screen; a deliberate blink (or Space) starts it."""
        if self.can_setup and self._setup is None:
            self.state.setup = {"phase": "ready"}

    def start_setup(self, t):
        if not self.can_setup:
            return
        self._setup = {"t0": t, "X": [], "y": []}
        self._setup_state(t)

    def cancel_setup(self):
        self._setup = None; self.state.setup = None

    def _setup_state(self, t):
        el = t - self._setup["t0"]; i = min(int(el // SETUP_HOLD_S), len(SETUP_ORDER) - 1)
        z = SETUP_ORDER[i]
        self.state.setup = {"phase": "run", "i": i, "n": len(SETUP_ORDER), "zone": z, "x": self.cfg.setup_x[z],
                            "p": round(min(1.0, (el - i * SETUP_HOLD_S) / SETUP_HOLD_S), 2)}

    def _setup_step(self, t, gf):
        su = self._setup; el = t - su["t0"]; i = int(el // SETUP_HOLD_S)
        if i >= len(SETUP_ORDER):
            return self._finish_setup(t)
        if gf is not None and el - i * SETUP_HOLD_S >= SETUP_SETTLE_S:
            su["X"].append(gf); su["y"].append("LCR".index(SETUP_ORDER[i]))
        self._setup_state(t)

    def _fit_from_setup(self, X, y):
        self.gm.fit(X, y)
        self.gm.zone_x = dict(self.cfg.setup_x)
        return round(float((self.gm.proba(X).argmax(1) == y).mean()), 3)

    def _finish_setup(self, t):
        su, self._setup = self._setup, None
        y = np.array(su["y"], int)
        counts = [int((y == k).sum()) for k in range(3)]
        if not self.adaptable:                          # new user: learn the zones from the setup itself
            if min(counts) >= 15:
                acc = self._fit_from_setup(np.array(su["X"]), y)
                info = {"ok": True, "changed": True, "fitted": True, "frames": int(len(y)), "train_acc": acc}
            else:
                info = {"ok": False, "reason": "eyes not seen at every dot"}
        elif sum(c >= 5 for c in counts) >= 2:
            X = np.array(su["X"])
            info = self.gm.recalibrate(X, y)
            acc = info.get("acc_after", info.get("acc_before", 1.0))
            if info.get("ok") and acc < REFIT_BELOW and min(counts) >= 15:     # aligning wasn't enough
                info.update(refit=True, changed=True, acc_aligned=acc, train_acc=self._fit_from_setup(X, y))
        else:
            info = {"ok": False, "reason": "eyes not seen at the dots"}
        if info.get("ok"):
            self.state.setup_ok, self.state.corrections = True, 0
            msg = ("Learned your eyes in 10 seconds" if info.get("fitted") else
                   "Re-learned your eyes (glasses or light changed?)" if info.get("refit") else
                   "Adjusted to how you are sitting today" if info.get("changed") else
                   "Checked: no adjustment needed")
        else:
            msg = "Setup failed (" + info.get("reason", "") + "). Press K to try again."
        self.gm.hmm.reset(); self.hold_zone, self.hold_since = "C", t
        self.block_until = t + self.cfg.refractory_s
        self.state.setup = {"phase": "done", "ok": bool(info.get("ok")), "msg": msg, "until": t + 3.0}
        self.events.append((t, "setup", info))
        self.on_setup(info)

    # ------------------------------------------------------------------ webcam null-zone finder
    def offer_nullfind(self):
        if self.can_setup and self._null is None and self._setup is None:
            self.state.nullfind = {"phase": "ready"}

    def start_nullfind(self, t):
        self._null = NullFinder(t, zone_targets(0.5))     # measured around the screen centre
        self.state.nullfind = self._null.state

    def cancel_nullfind(self):
        self._null = None; self.state.nullfind = None

    def _nullfind_step(self, t, g):
        self.state.nullfind = self._null.step(t, g)
        if not self._null.done:
            return
        res, self._null = self._null.result, None
        if res.get("ok"):
            n = res["null_deg"]
            msg = (f"Calmest gaze: {abs(n):.0f}° {'right' if n > 0 else 'left'} of centre" if res["clear"]
                   else f"No clear null zone away from the centre (eyes settled at {n:+.0f}°)")
            if res.get("agree") is False:
                msg += " (head and eyes disagree: treat with care)"
        else:
            msg = "Couldn't measure (" + res.get("reason", "") + "). Press N to try again."
        extra = self.on_nullzone(res) or ""
        self.gm.hmm.reset(); self.hold_zone, self.hold_since = "C", t
        self.block_until = t + self.cfg.refractory_s
        self.state.nullfind = {"phase": "done", "ok": bool(res.get("ok")), "msg": f"{msg}. {extra}" if extra else msg,
                               "until": t + 6.0, "result": res}
        self.events.append((t, "nullzone", res))

    # ------------------------------------------------------------------ phrase-selection test
    def offer_task(self):
        from nullpoint.board.task import TaskRunner
        if len(PAGES[0][1]) < 4:
            return                                # the test needs at least 2 phrases on the first page
        if self.model_ready and self._setup is None and self._null is None and (self.task is None or not self.task.active):
            self.task = TaskRunner(sim_amp=self.sim_amp, adapt_before=self.cfg.adapt)
            self.state.task = self.task.state

    def cancel_task(self):
        if self.task is not None:
            self.set_adapt(self.task.adapt_before)
        self.task = None; self.state.task = None

    def _self_correct(self, t):
        """Free label: just before a deliberate blink the user was reading the focus card (zone C)."""
        if not (self.cfg.self_correct and self.adaptable and self.buf) or self.state.emergency:
            return                                 # (a blink that clears an alarm says nothing about gaze)
        t_end = self.buf[-1][0]                    # last frame before the eyelids started closing
        if t - t_end > 2.0:
            return
        lo, hi = t_end - CORRECT_WIN[0], t_end - CORRECT_WIN[1]
        X = np.array([g for (tt, g) in self.buf if lo <= tt <= hi])
        step = self.gm.nudge(X, 1) if len(X) else None
        if step is not None:
            self.state.corrections += 1
            self.events.append((t, "self_correct", step))

    # ------------------------------------------------------------------ per frame
    @staticmethod
    def _run(filters, g):
        gf = g.copy()
        for j, flt in zip(EYE_DIMS, filters):
            gf[j] = flt(float(g[j]))
        return gf

    def update(self, t: float, g: np.ndarray | None, b: np.ndarray | None):
        st = self.state
        st.face = g is not None
        if st.setup and st.setup.get("phase") == "done" and t > st.setup["until"]:
            st.setup = None
        if st.nullfind and st.nullfind.get("phase") == "done" and t > st.nullfind["until"]:
            st.nullfind = None
        g = self._inject(t, g)
        kind, info = self.bm.step(t, b) if b is not None else (None, {})
        if kind == "emergency_pending":
            st.alarm_pending, kind = True, None
        elif kind == "rest":
            st.alarm_pending, kind = False, None
            self.events.append((t, "rest", info.get("dur")))
        closing = b is not None and (b[0] + b[1]) / 2 > 0.6 * BLINK_ON
        gf = None
        if g is not None and not closing:                   # glance frozen during blinks
            gp, ga = self._run(self.f_plain, g), self._run(self.f_adapt, g)
            gf = ga if self.cfg.adapt else gp
            st.strip = self._strip(gp, ga) if self.model_ready else None
        if self.task is not None:
            self.task.tick(t, self)

        if self._setup is not None:                     # setup running: collect, don't act
            if kind == "emergency":
                self.emergency(t)
            else:
                self._setup_step(t, gf)
            return st
        if self._null is not None:                      # null-zone finder running: same rules
            if kind == "emergency":
                self.emergency(t)
            else:
                self._nullfind_step(t, g if (g is not None and not closing) else None)
            return st
        null_ready = bool(st.nullfind and st.nullfind.get("phase") == "ready")
        task_ready = bool(self.task is not None and self.task.waiting)
        ready = bool(st.setup and st.setup.get("phase") == "ready") or null_ready or task_ready

        if self.quiet and kind == "deliberate":
            kind = None                                  # the game uses gaze only (alarms still work)
        if kind == "deliberate" and self.task is not None and self.task.finished:
            self.cancel_task(); return st                # results read: close them
        if kind == "deliberate":
            st.blink_p = info.get("p")
            if null_ready:
                self.start_nullfind(t); return st
            if task_ready:
                self.task.start(t, self); return st
            if ready:
                self.start_setup(t); return st
            self._self_correct(t)
            self.select(t)
        elif kind == "natural":
            st.blink_p = info.get("p"); self.events.append((t, "natural_blink", info.get("p")))
        elif kind == "emergency":
            self.emergency(t)
        if gf is None or not self.model_ready:          # no model yet: only the setup can start
            return st
        self.buf.append((t, gf))
        post = self.gm.proba(gf)[0]
        belief = self.gm.hmm.step(post)
        st.belief = belief.tolist()
        z = "LCR"[int(np.argmax(belief))] if belief.max() >= self.cfg.belief_on else "C"
        st.zone = z
        if self.quiet or ready or t < self.block_until or (self.task is not None and self.task.paused):
            self.hold_zone = "C"; return st
        if z != self.hold_zone:
            self.hold_zone, self.hold_since, self.hold_steps = z, t, 0
        elif z in "LR" and self.hold_steps < MAX_REPEAT:
            held = t - self.hold_since
            if held >= self.dwell and t - self.last_step >= (self.cfg.repeat_s if self.last_step > self.hold_since else 0):
                self._step(-1 if z == "L" else 1, t)
                self.hold_steps += 1
        return st
