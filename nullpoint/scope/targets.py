"""Fixation targets for slow-mo recording sessions.

Shows a dot at known horizontal gaze angles (from seating distance + screen width), holding
each for HOLD seconds, with a short beep at every change, so the phone video can be split
into gaze holds afterwards (the saccade between holds is easy to find in the eye trace).

    python -m nullpoint.scope.targets                # default -15..+15 deg in 5 steps
    python -m nullpoint.scope.targets --angles -15 -7.5 0 7.5 15 --hold 6

Order is always: centre first, then the listed angles, then centre again (drift check).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import time
import winsound

import cv2
import numpy as np

from nullpoint.config import deg_to_screen_x, RAW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=float, nargs="+", default=[-15, -7.5, 7.5, 15])
    ap.add_argument("--hold", type=float, default=6.0)
    a = ap.parse_args()
    u = ctypes.windll.user32
    SW, SH = u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    seq = [0.0] + list(a.angles) + [0.0]
    for ang in seq:
        fx = deg_to_screen_x(ang)
        if not 0.02 < fx < 0.98:
            raise SystemExit(f"{ang} deg is off-screen at this seating distance (x={fx:.2f})")
    cv2.namedWindow("targets", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("targets", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("targets", cv2.WND_PROP_TOPMOST, 1)

    def draw(dot, msg, sub=""):
        img = np.full((SH, SW, 3), 25, np.uint8)
        if dot is not None:
            c = (int(dot * SW), SH // 2)
            cv2.circle(img, c, 26, (255, 255, 255), -1); cv2.circle(img, c, 6, (0, 0, 230), -1)
        cv2.putText(img, msg, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (210, 210, 210), 2)
        if sub:
            cv2.putText(img, sub, (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        cv2.imshow("targets", img)
        return cv2.waitKey(1) & 0xFF

    while True:
        k = draw(None, "Start the slow-mo recording, keep your head still, then press SPACE",
                 f"targets: {', '.join(f'{v:+g} deg' for v in seq)}  -  {a.hold:g} s each  (Esc = quit)")
        if k == 27:
            return
        if k == 32:
            break
    log = []
    t0 = time.perf_counter()
    for i, ang in enumerate(seq):
        winsound.Beep(1200 if i == 0 else 900, 120)
        log.append({"t": round(time.perf_counter() - t0, 3), "deg": ang})
        ts = time.perf_counter()
        while time.perf_counter() - ts < a.hold:
            if draw(deg_to_screen_x(ang), f"Look at the red centre   ({i + 1}/{len(seq)})", f"{ang:+g} deg") == 27:
                return
    winsound.Beep(600, 300)
    draw(None, "Done - stop the recording"); cv2.waitKey(1200); cv2.destroyAllWindows()
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"targets-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"hold": a.hold, "sequence": log}, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
