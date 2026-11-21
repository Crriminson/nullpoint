"""Shared constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CALIB = DATA / "calib"
PROFILES = DATA / "profiles"
REPORTS = ROOT / "reports"
MODEL_FILE = ROOT / "models" / "face_landmarker.task"
TRAINED = DATA / "trained"   # per-user fitted models (personal, git-ignored)

# anatomy (population averages; stated as assumptions in the report)
IRIS_DIAMETER_MM = 11.7      # horizontal visible iris diameter
EYE_ROTATION_RADIUS_MM = 12.0  # distance from centre of rotation to the iris plane, approx.

# setup (measured by the user)
SEAT_DISTANCE_CM = 55.0      # eye to screen, 50-60 cm
SCREEN_WIDTH_CM = 34.4       # 15.6" 16:9 panel (Vivobook 15)
SCREEN_HEIGHT_CM = 19.4

# MediaPipe landmark ids (subject's right eye = image left)
R_OUT, R_IN, R_IRIS, R_TOP, R_BOT = 33, 133, 468, 159, 145
L_OUT, L_IN, L_IRIS, L_TOP, L_BOT = 263, 362, 473, 386, 374
NOSE_TIP = 1


def screen_x_to_deg(frac: float) -> float:
    """Horizontal gaze angle (deg) of a point at screen-width fraction `frac` (0.5 = centre)."""
    import math
    return math.degrees(math.atan((frac - 0.5) * SCREEN_WIDTH_CM / SEAT_DISTANCE_CM))


def deg_to_screen_x(deg: float) -> float:
    import math
    return 0.5 + math.tan(math.radians(deg)) * SEAT_DISTANCE_CM / SCREEN_WIDTH_CM


def null_screen_x(profile: dict | None) -> float:
    """Screen x (0..1) of the focus card: the null-point angle if the profile has one, else centre.
    Clamped so both side zones keep room. Shared by calibration and the board so the trained
    centre zone and the displayed focus card coincide."""
    if profile and profile.get("null_deg") is not None:
        return min(max(deg_to_screen_x(profile["null_deg"]), 0.3), 0.7)
    return 0.5


def zone_targets(c: float = 0.5) -> dict:
    """Screen x (0..1) of the L / C / R gaze targets around the focus position c. Shared by the
    full calibration and the board's 10-second setup so both train on the same positions."""
    return {"L": max(0.08, c - 0.33), "C": c, "R": min(0.92, c + 0.33)}


def load_profile(path=None) -> dict:
    import json
    p = Path(path) if path else PROFILES / "profile.json"
    return json.loads(p.read_text()) if p.exists() else {}
