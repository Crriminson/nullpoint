"""Feasibility probe: guided gaze scan.

Shows a dot on a 3x5 grid (4 s each, centre first). Records the webcam at 1280x720/30fps
(MSMF) to data/raw/ and the per-frame iris/eyelid signals to an .npz next to it.
Press Esc to abort. Everything stays local (data/ is git-ignored).
"""
import time, pathlib, ctypes, argparse
import numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"; OUT.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--cam", type=int, default=0); ap.add_argument("--label", default="")
args = ap.parse_args()
stamp = time.strftime("%Y%m%d-%H%M%S") + (f"-{args.label}" if args.label else "")
XS, YS = [0.1, 0.3, 0.5, 0.7, 0.9], [0.5, 0.15, 0.85]
POS = [(0.5, 0.5)] + [(x, y) for y in YS for x in XS if (x, y) != (0.5, 0.5)]  # centre first, then rows
HOLD = 4.0

user32 = ctypes.windll.user32
SW, SH = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

cap = cv2.VideoCapture(args.cam, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080); cap.set(cv2.CAP_PROP_FPS, 30)
FW, FH = int(cap.get(3)), int(cap.get(4)); print("camera", args.cam, FW, FH)
writer = cv2.VideoWriter(str(OUT / f"scan-{stamp}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (FW, FH))

lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(ROOT / "models" / "face_landmarker.task")),
    running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))

cv2.namedWindow("scan", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("scan", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setWindowProperty("scan", cv2.WND_PROP_TOPMOST, 1)

def screen(dot, msg):
    img = np.full((SH, SW, 3), 30, np.uint8)
    if dot is not None:
        c = (int(dot[0] * SW), int(dot[1] * SH))
        cv2.circle(img, c, 22, (255, 255, 255), -1); cv2.circle(img, c, 5, (0, 0, 255), -1)
    cv2.putText(img, msg, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 200), 2)
    cv2.imshow("scan", img)
    return cv2.waitKey(1) & 0xFF

# wait for the user: live preview + press SPACE to start (Esc aborts)
while True:
    ok, frame = cap.read()
    img = np.full((SH, SW, 3), 30, np.uint8)
    if ok:
        pv = cv2.resize(frame, (SW // 3, SW // 3 * FH // FW)); img[SH - pv.shape[0] - 20:SH - 20, 20:20 + pv.shape[1]] = pv
    cv2.putText(img, "Head still, eyes follow the dot. Press SPACE to start (Esc = quit)", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
    cv2.imshow("scan", img); k = cv2.waitKey(1) & 0xFF
    if k == 27: raise SystemExit
    if k == 32: break
t_end = time.perf_counter() + 3
while time.perf_counter() < t_end:
    screen(None, f"Starting in {t_end - time.perf_counter():.0f}...")
rows = []  # t, target, landmarks(478x3 flattened), blink L, blink R
t0 = time.perf_counter()
for k, x in enumerate(POS):
    t_seg = time.perf_counter()
    while time.perf_counter() - t_seg < HOLD:
        if screen(x, f"Look at the red centre  ({k + 1}/{len(POS)})") == 27:
            raise SystemExit
        ok, frame = cap.read()
        if not ok:
            continue
        t = time.perf_counter() - t0
        writer.write(frame)
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(t * 1000))
        if r.face_landmarks:
            L = np.array([[p.x, p.y, p.z] for p in r.face_landmarks[0]], np.float32)
            bs = {b.category_name: b.score for b in r.face_blendshapes[0]}
            rows.append((t, x, L, bs["eyeBlinkLeft"], bs["eyeBlinkRight"]))
        else:
            rows.append((t, x, np.full((478, 3), np.nan, np.float32), np.nan, np.nan))
screen(None, "Done - thank you!"); cv2.waitKey(800)
cap.release(); writer.release(); cv2.destroyAllWindows()

np.savez_compressed(OUT / f"scan-{stamp}.npz",
                    t=np.array([r[0] for r in rows]), target=np.array([r[1][0] for r in rows]), target_y=np.array([r[1][1] for r in rows]),
                    lms=np.stack([r[2] for r in rows]), blinkL=np.array([r[3] for r in rows]),
                    blinkR=np.array([r[4] for r in rows]), frame_wh=np.array([FW, FH]), screen_wh=np.array([SW, SH]))
print("saved", OUT / f"scan-{stamp}.npz", len(rows), "frames")
