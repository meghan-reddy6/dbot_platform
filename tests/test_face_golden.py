"""The face pipeline on a real face with this machine's ONNX Runtime and session settings (no ultralytics needed).

CI runs this file on x86 and on 64-bit ARM. On the Snapdragon QCS8550 board these crops pass both with DeskBot's
settings and with ONNX Runtime's fused int8 defaults: this file catches gross breakage, but not the fused-kernel
error that create_session works around on 64-bit ARM.
"""
from pathlib import Path

import cv2
import pytest

# US Navy portrait of Rear Admiral Grace M. Hopper (public domain), copied from matplotlib's sample data.
IMAGE = Path(__file__).parent / "assets" / "grace_hopper.jpg"
HEAD = (67, 32, 468, 433)  # head crop the pose model gives for this photo (x1, y1, x2, y2)


def _crop(dx=0.0, dy=0.0, scale=1.0):
    image = cv2.imread(str(IMAGE))
    x1, y1, x2, y2 = HEAD
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    cx, cy = (x1 + x2) / 2 + dx * (x2 - x1), (y1 + y2) / 2 + dy * (y2 - y1)
    a, b = int(max(0, cx - w / 2)), int(max(0, cy - h / 2))
    return image[b:int(min(image.shape[0], cy + h / 2)), a:int(min(image.shape[1], cx + w / 2))].copy()


@pytest.fixture(scope="module")
def recognizer():
    from recognition.face_recognizer import FaceRecognizer

    return FaceRecognizer()


@pytest.mark.parametrize("dx, dy, scale", [(0, 0, 1.0), (0.08, 0, 1.0), (-0.08, 0, 1.0), (0, 0.08, 1.0),
                                           (0, -0.08, 1.0), (0, 0, 0.85), (0, 0, 1.2)])
def test_face_is_found_and_embedded_consistently(recognizer, dx, dy, scale):
    reference = recognizer.extract_embedding(_crop())
    embedding = recognizer.extract_embedding(_crop(dx, dy, scale))
    assert reference is not None and embedding is not None
    assert float(reference @ embedding) > 0.8  # both are L2-normalized: the same person from a shifted crop


def test_background_without_a_face_is_rejected(recognizer):
    image = cv2.imread(str(IMAGE))
    assert recognizer.extract_embedding(image[440:600, 0:160].copy()) is None
