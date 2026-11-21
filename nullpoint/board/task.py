"""Phrase-selection test for the end-to-end evaluation (report §4.9). Press T on the board.

The board asks for a target phrase on the first page; the user moves to it with glances and selects
it with a deliberate blink. Four blocks of six phrases, Adapt off / on in ABBA or BAAB order (chosen
at random), with whatever simulated nystagmus is switched on (S) recorded alongside.

Per phrase: time from prompt to the correct selection, wrong selections on the way, and steps
beyond the shortest route. Every phrase starts with the cursor on the first tile and the targets of
each block are tiles 1-6 in shuffled order, so every block has the same difficulty (shortest routes
of 1, 2, 3, 4, 3 and 2 steps). A phrase not found within 30 s counts as a failure. The block's
condition is not shown on screen, though the board's behaviour can give it away (not blinded).
"""
from __future__ import annotations

import random
import time

import numpy as np

from nullpoint.board.phrases import PAGES

TARGETS = [1, 2, 3, 4, 5, 6]          # default page 0 (7 phrases + "More options")
BLOCK_PAUSE_S, GAP_S, TIMEOUT_S = 4.0, 1.2, 30.0


def min_steps(tile: int, n: int) -> int:
    return min(tile, n - tile)


def summarize(trials: list) -> dict:
    out = {}
    for cond in ("off", "on"):
        tr = [x for x in trials if x["adapt"] == cond]
        ok = [x for x in tr if x["found"]]
        out[cond] = {
            "phrases": len(tr), "found": len(ok),
            "median_time_s": round(float(np.median([x["time_s"] for x in ok])), 2) if ok else None,
            "wrong_selections_per_phrase": round(float(np.mean([x["wrong"] for x in tr])), 2) if tr else None,
            "extra_steps_per_phrase": round(float(np.mean([x["steps"] - x["min_steps"] for x in tr])), 2) if tr else None,
        }
    return out


class TaskRunner:
    def __init__(self, sim_amp: float = 0.0, adapt_before: bool = True, seed: int | None = None):
        rng = random.Random(seed if seed is not None else time.time_ns())
        self.order = rng.choice([["off", "on", "on", "off"], ["on", "off", "off", "on"]])
        n = len(PAGES[0][1])                   # targets: every tile but the first and "More options"
        targets = TARGETS if n == 8 else list(range(1, n - 1))
        self.n_targets = len(targets)
        self.blocks = [rng.sample(targets, len(targets)) for _ in self.order]
        self.sim_amp, self.adapt_before = sim_amp, adapt_before
        self.phase, self.bi, self.ti, self.t_phase = "ready", 0, -1, 0.0
        self.trials, self.cur, self.summary, self.saved = [], None, None, ""
        self.key_used = False

    # ---------------------------------------------------------------- flags read by the selector
    @property
    def waiting(self):
        return self.phase == "ready"

    @property
    def active(self):
        return self.phase in ("block", "trial", "gap")

    @property
    def paused(self):
        return self.phase in ("ready", "block", "gap", "done")

    @property
    def finished(self):
        return self.phase == "done"

    @property
    def state(self):
        st = {"phase": self.phase, "block": self.bi + 1, "blocks": len(self.order),
              "trial": max(self.ti, 0) + 1, "trials": self.n_targets, "sim_amp": self.sim_amp}
        if self.phase == "trial":
            st["target"] = self.cur["target_text"]
        if self.phase == "gap" and self.cur:
            st["last"] = "found" if self.cur["found"] else "timeout"
        if self.phase == "done":
            st.update(summary=self.summary, saved=self.saved, order=self.order)
        return st

    # ---------------------------------------------------------------- flow
    def start(self, t, sel):
        self.phase, self.bi, self.ti, self.t_phase = "block", 0, -1, t
        sel.set_adapt(self.order[0] == "on")
        sel.state.task = self.state

    def _prompt(self, t, sel):
        self.ti += 1
        tiles = PAGES[0][1]
        tgt = self.blocks[self.bi][self.ti]
        sel.state.page, sel.state.cursor = 0, 0
        getattr(sel.gm.hmm, "reset", lambda: None)()
        self.cur = {"block": self.bi + 1, "adapt": self.order[self.bi], "target": tgt, "target_text": tiles[tgt],
                    "min_steps": min_steps(tgt, len(tiles)), "t0": t, "steps": 0, "wrong": 0, "found": False,
                    "time_s": None}
        self.phase, self.t_phase = "trial", t

    def _end_trial(self, t, found):
        self.cur.update(found=found, time_s=round(t - self.cur["t0"], 3) if found else None)
        self.trials.append({k: v for k, v in self.cur.items() if k != "t0"})
        self.phase, self.t_phase = "gap", t

    def tick(self, t, sel):
        if self.phase == "block" and t - self.t_phase >= BLOCK_PAUSE_S:
            self._prompt(t, sel)
        elif self.phase == "trial" and t - self.cur["t0"] >= TIMEOUT_S:
            self._end_trial(t, False)
        elif self.phase == "gap" and t - self.t_phase >= GAP_S:
            if self.ti + 1 < len(self.blocks[self.bi]):
                self._prompt(t, sel)
            elif self.bi + 1 < len(self.order):
                self.bi, self.ti, self.phase, self.t_phase = self.bi + 1, -1, "block", t
                sel.set_adapt(self.order[self.bi] == "on")
            else:
                self._finish(sel)
        sel.state.task = self.state

    def on_step(self, t, d, source):
        if self.phase == "trial":
            self.cur["steps"] += 1
            self.key_used |= source != "eyes"

    def on_select(self, t, tile, source, sel):
        if self.phase != "trial":
            return
        self.key_used |= source != "eyes"
        if tile == self.cur["target_text"]:
            self._end_trial(t, True)
        else:
            self.cur["wrong"] += 1

    def _finish(self, sel):
        self.phase = "done"
        self.summary = summarize(self.trials)
        meta = {"order": self.order, "sim_amp_deg": self.sim_amp, "keyboard_used": self.key_used,
                "when": time.strftime("%Y-%m-%d %H:%M")}
        self.saved = sel.on_task(self.summary | {"meta": meta}, self.trials) or ""
        sel.set_adapt(self.adapt_before)
