"""Guided calibration session -> labelled training data for the glance and blink models.

    python -m nullpoint.calibrate --label A          (about 3 minutes, fullscreen)

Blocks
  1. glance   : dot at the board's zone centres (L / C / R), random order, 4 x 2.5 s each
  2. blink    : 12 cues "blink firmly (hold it a moment)" at random 3-5 s intervals
  3. natural  : 75 s of reading a paragraph; every blink here is a natural (involuntary) blink
Saves data/calib/calib-<stamp>-<label>.npz with per-frame landmarks (float16), blendshapes,
pose matrix, and the block/target/cue labels. Raw video is NOT saved.
"""
from __future__ import annotations

import argparse
import ctypes
import random
import time
import winsound

import cv2
import numpy as np

from nullpoint.capture import Camera
from nullpoint.config import CALIB, null_screen_x, load_profile, zone_targets

ZONES = {"L": 0.17, "C": 0.5, "R": 0.83}     # must match the board layout (board/static/board.js)
BLEND_KEYS = None

READING = ("Nullpoint is a communication board for people whose eyes do not hold still. "
           "It measures the rhythm of the eyes first, and then it uses that rhythm to read "
           "glances and blinks more reliably. Please read this text at your normal pace, "
           "and blink whenever you naturally would. There is nothing you need to do with the "
           "board right now: this part only records how your eyes behave when you are not "
           "trying to select anything. The quick brown fox jumps over the lazy dog. "
           "Monsoon clouds gather over the Arabian Sea while trains rattle along the coast. "
           "If you reach the end, start again from the top and keep reading until the timer ends.")


class Recorder:
    def __init__(self):
        self.rows = []
        self.block, self.target, self.cue = "idle", "", 0

    def __call__(self, ff):
        global BLEND_KEYS
        if ff.lms is None:
            self.rows.append((ff.t, None, None, None, self.block, self.target, self.cue, ff.brightness)); return
        if BLEND_KEYS is None and ff.blend:
            BLEND_KEYS = sorted(ff.blend)
        bv = np.array([ff.blend.get(k, 0.0) for k in BLEND_KEYS], np.float32) if ff.blend else None
        self.rows.append((ff.t, ff.lms.astype(np.float16), bv, ff.pose.astype(np.float32) if ff.pose is not None else None,
                          self.block, self.target, self.cue, ff.brightness))
        self.cue = 0


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--short", action="store_true", help="quick test run (fewer trials)")
    ap.add_argument("--no-adapt", action="store_true", help="ignore the nystagmus profile (centre zone at screen centre)")
    a = ap.parse_args()
    if not a.no_adapt:
        ZONES.update(zone_targets(null_screen_x(load_profile())))
    u = ctypes.windll.user32
    SW, SH = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    rec = Recorder()
    cam = Camera(a.cam); cam.on_frame(rec); cam.start()
    cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("calib", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("calib", cv2.WND_PROP_TOPMOST, 1)
    F = cv2.FONT_HERSHEY_SIMPLEX

    def show(msg="", dot=None, sub="", lines=None, color=(235, 235, 235)):
        img = np.full((SH, SW, 3), 28, np.uint8)
        if dot is not None:
            c = (int(dot * SW), SH // 2)
            cv2.circle(img, c, 30, (255, 255, 255), -1); cv2.circle(img, c, 7, (0, 0, 230), -1)
        if msg:
            cv2.putText(img, msg, (50, 80), F, 1.2, color, 2)
        if sub:
            cv2.putText(img, sub, (50, 130), F, 0.8, (160, 160, 160), 2)
        if lines:
            for i, ln in enumerate(lines):
                cv2.putText(img, ln, (90, 240 + i * 52), F, 1.0, (230, 230, 230), 2)
        jp = cam.latest_jpeg()
        if jp is not None and dot is None and lines is None:
            pv = cv2.imdecode(np.frombuffer(jp, np.uint8), cv2.IMREAD_COLOR)
            img[SH - pv.shape[0] - 20:SH - 20, 20:20 + pv.shape[1]] = pv
        cv2.imshow("calib", img)
        return cv2.waitKey(1) & 0xFF

    def wait_space(msg, sub):
        while True:
            last = rec.rows[-1] if rec.rows else None
            face = "face OK" if last is not None and last[1] is not None else "NO FACE - adjust camera/light"
            k = show(msg, sub=f"{sub}   [{face}, {cam.fps:.0f} fps]")
            if k == 27:
                raise SystemExit("aborted")
            if k == 32:
                return

    def hold(seconds, **kw):
        t = time.perf_counter()
        while time.perf_counter() - t < seconds:
            if show(**kw) == 27:
                raise SystemExit("aborted")

    try:
        wait_space("Calibration: sit as you will use the board. SPACE to start.", "Esc quits")
        # 1. glance
        seq = [z for z in "LCR" for _ in range(2 if a.short else 4)]
        random.shuffle(seq)
        rec.block = "glance"
        for i, z in enumerate(seq):
            rec.target = z; winsound.Beep(1000, 60)
            hold(2.5, dot=ZONES[z], msg=f"Look at the dot  ({i + 1}/{len(seq)})")
        rec.block, rec.target = "idle", ""
        # 2. deliberate blinks
        wait_space("Next: when the screen says BLINK, blink firmly - hold it shut for a moment.", "SPACE to start")
        rec.block = "blink"
        for i in range(6 if a.short else 12):
            hold(random.uniform(3.0, 5.0), msg="Look at the centre and relax", dot=0.5)
            rec.cue = 1; winsound.Beep(1400, 80)
            hold(1.8, msg="BLINK", sub=f"({i + 1})", color=(80, 220, 255))
        hold(2.0, msg="Relax", dot=0.5)
        rec.block = "idle"
        # 3. natural blinks while reading
        wait_space("Last part: read the text normally for 75 seconds. Blink as you normally would.", "SPACE to start")
        rec.block = "natural"
        lines = _wrap(READING, 70)
        hold(30 if a.short else 75, lines=lines, msg="Read normally", sub="")
        rec.block = "idle"
        show("Done - thank you!"); cv2.waitKey(800)
    finally:
        cam.stop(); cv2.destroyAllWindows()
    CALIB.mkdir(parents=True, exist_ok=True)
    out = CALIB / f"calib-{time.strftime('%Y%m%d-%H%M%S')}{'-' + a.label if a.label else ''}.npz"
    R = rec.rows
    has = np.array([r[1] is not None for r in R])
    lms = np.stack([r[1] if r[1] is not None else np.full((478, 3), np.nan, np.float16) for r in R])
    nb = len(BLEND_KEYS or [])
    blend = np.stack([r[2] if r[2] is not None else np.full(nb, np.nan, np.float32) for r in R]) if nb else np.zeros((len(R), 0))
    pose = np.stack([r[3] if r[3] is not None else np.full((4, 4), np.nan, np.float32) for r in R])
    np.savez_compressed(out, t=np.array([r[0] for r in R]), has_face=has, lms=lms, blend=blend,
                        blend_keys=np.array(BLEND_KEYS or []), pose=pose, block=np.array([r[4] for r in R]),
                        target=np.array([r[5] for r in R]), cue=np.array([r[6] for r in R]),
                        brightness=np.array([r[7] for r in R]), wh=np.array([1280, 720]))
    print("saved", out, len(R), "frames, face in", f"{has.mean():.0%}")


if __name__ == "__main__":
    main()
