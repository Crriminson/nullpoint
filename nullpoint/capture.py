"""Live webcam capture + MediaPipe Face Landmarker in a background thread.

Built-in camera via Media Foundation (MSMF) gives 1280x720 @ 30 fps on the target laptop;
DirectShow is stuck at 10 fps there.
"""
from __future__ import annotations

import threading
from pathlib import Path
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

from nullpoint.config import MODEL_FILE


@dataclass
class FaceFrame:
    t: float                    # seconds (perf_counter based)
    lms: np.ndarray | None      # (478,3) normalised landmarks, None if no face
    blend: dict | None          # blendshape name -> score
    pose: np.ndarray | None     # 4x4 facial transformation matrix
    wh: tuple[int, int]
    brightness: float


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/"
             "face_landmarker.task")


def ensure_model(path=MODEL_FILE, url=MODEL_URL, urlopen=None):
    """MediaPipe's face model (Apache 2.0, ~3.6 MB) is downloaded on first run, not committed."""
    path = Path(path)
    if path.exists():
        return path
    import shutil
    import urllib.request
    urlopen = urlopen or urllib.request.urlopen
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"first run: downloading the MediaPipe face model to {path} ...")
    try:
        with urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if tmp.stat().st_size < 1_000_000:
            raise IOError("the download is too small to be the model")
        tmp.replace(path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"couldn't download the face model ({e}). Download it once by hand:\n"
                         f"  curl -L -o models/face_landmarker.task {url}")
    return path


def make_landmarker(mode=vision.RunningMode.VIDEO):
    ensure_model()
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_FILE)), running_mode=mode, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True,
        min_face_detection_confidence=0.5, min_tracking_confidence=0.5))


def open_camera(index: int = 0, w: int = 1280, h: int = 720, fps: int = 30):
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h); cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cap


def detect(landmarker, frame_bgr: np.ndarray, t: float) -> FaceFrame:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    r = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(t * 1000))
    h, w = frame_bgr.shape[:2]
    br = float(frame_bgr[::8, ::8].mean())
    if not r.face_landmarks:
        return FaceFrame(t, None, None, None, (w, h), br)
    L = np.array([[p.x, p.y, p.z] for p in r.face_landmarks[0]], np.float32)
    bs = {b.category_name: b.score for b in r.face_blendshapes[0]} if r.face_blendshapes else {}
    pose = np.array(r.facial_transformation_matrixes[0]) if r.facial_transformation_matrixes else None
    return FaceFrame(t, L, bs, pose, (w, h), br)


class Camera:
    """Background capture + landmarking. Subscribers get every FaceFrame via callbacks;
    `latest_jpeg()` gives a small preview for the UI."""

    def __init__(self, index: int = 0, source=None):
        self.index, self.source = index, source
        self.callbacks = []
        self.frames = deque(maxlen=300)
        self._stop = threading.Event()
        self._thread = None
        self._jpeg = None
        self.fps = 0.0
        self.t0 = time.perf_counter()   # frame clock: FaceFrame.t = perf_counter() - t0

    def on_frame(self, cb):
        self.callbacks.append(cb)

    def start(self):
        self.t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True); self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def latest_jpeg(self):
        return self._jpeg

    def _run(self):
        cap = cv2.VideoCapture(str(self.source)) if self.source is not None else open_camera(self.index)
        lm = make_landmarker()
        t0 = self.t0; n = 0; tl = time.perf_counter(); last_ms = -1
        file_dt = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 30.0) if self.source is not None else 0.0
        next_t = time.perf_counter()
        while not self._stop.is_set():
            if file_dt:                    # a recorded clip: play it in real time, on a loop
                next_t += file_dt
                time.sleep(max(0.0, next_t - time.perf_counter()))
            ok, frame = cap.read()
            if not ok:
                if file_dt:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01); continue
            t = time.perf_counter() - t0
            if int(t * 1000) <= last_ms:   # mediapipe needs strictly increasing timestamps
                continue
            last_ms = int(t * 1000)
            ff = detect(lm, frame, t)
            self.frames.append(ff)
            for cb in self.callbacks:
                try:
                    cb(ff)
                except Exception as e:   # never let a consumer kill capture
                    print("frame callback error:", e)
            n += 1
            if n % 3 == 0:
                small = cv2.resize(frame, (320, 180))
                self._jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            if t - (tl - t0) >= 1.0:
                self.fps = n / (t - (tl - t0)); n = 0; tl = time.perf_counter()
        cap.release()
