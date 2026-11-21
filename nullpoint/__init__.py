"""Nullpoint: an eye-communication board that measures nystagmus and adapts to it."""
import os

# OpenCV's Media Foundation backend takes ~30 s to open the camera on this laptop unless
# hardware transforms are disabled (must be set before cv2 is first imported).
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
