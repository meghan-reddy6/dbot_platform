import json
from pathlib import Path

import numpy as np
import pytest

from core.tracking.geometry import Keypoint, head_crop_box
from recognition.face_recognizer import _DEFAULT_QUANTIZATION, FaceRecognizer, load_quantization

MODELS = Path(__file__).resolve().parents[1] / "models"


def test_quantization_comes_from_model_metadata(tmp_path):
    shipped = load_quantization(str(MODELS / "metadata.json"))
    metadata = json.loads((MODELS / "metadata.json").read_text(encoding="utf-8"))
    landmarks = metadata["model_files"]["face_landmark_detector.onnx"]["outputs"]["landmarks"]["quantization_parameters"]
    assert shipped["landmarks"] == (landmarks["scale"], landmarks["zero_point"])

    edited = json.loads(json.dumps(metadata))
    edited["model_files"]["face_detector.onnx"]["outputs"]["box_scores_2"]["quantization_parameters"] = {
        "scale": 0.5,
        "zero_point": 7,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(edited), encoding="utf-8")
    assert load_quantization(str(tmp_path / "metadata.json"))["box_scores_2"] == (0.5, 7)
    assert load_quantization(str(tmp_path / "missing.json")) == _DEFAULT_QUANTIZATION


def test_head_crop_covers_the_whole_head():
    w, h = 640, 480
    pose = {
        "nose": Keypoint(320, 150, w, h),
        "left_shoulder": Keypoint(400, 240, w, h),
        "right_shoulder": Keypoint(240, 240, w, h),
    }
    x1, y1, x2, y2 = head_crop_box(pose, w, h)
    assert (x1, x2) == (240, 400)
    assert y1 < 150 - 60 and y2 > 150 + 60  # well above the eyes, well below the chin
    assert head_crop_box({**pose, "nose": Keypoint(320, 10, w, h)}, w, h)[1] == 0


@pytest.fixture(scope="module")
def recognizer():
    return FaceRecognizer()


def test_non_face_regions_are_not_embedded(recognizer):
    # Before the landmark presence gate, noise and background crops produced "face" embeddings.
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 255, (300, 200, 3), dtype=np.uint8) for _ in range(5)]
    crops.append(np.full((300, 200, 3), 128, np.uint8))
    assert all(recognizer.extract_embedding(crop) is None for crop in crops)
    assert recognizer.consecutive_errors == 0


def test_real_face_in_head_crop_is_embedded(recognizer):
    ultralytics = pytest.importorskip("ultralytics")
    import cv2

    from detection.person_detector import PersonDetector

    image = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"))
    h, w = image.shape[:2]
    detector = PersonDetector(str(MODELS / "yolov8n-pose.onnx"), providers=["CPUExecutionProvider"])
    embeddings = []
    for det in detector.detect(image):
        x1, y1, x2, y2 = head_crop_box(det["pose"], w, h)
        embeddings.append(recognizer.extract_embedding(image[y1:y2, x1:x2]))
    found = [e for e in embeddings if e is not None]
    assert found, "no face embedded from any head crop"
    assert all(np.linalg.norm(e) == pytest.approx(1.0, abs=1e-3) for e in found)
