import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from detection.person_detector import (
    HEAD_MODEL_POINTS,
    KEYPOINT_CONF,
    PersonDetector,
    decode_pose_output,
    letterbox,
    nms,
    solve_head_pose,
)

MODEL = Path(__file__).resolve().parents[1] / "models" / "yolov8n-pose.onnx"
FRAME_SHAPE = (480, 640)
CAMERA = np.array([[640.0, 0, 320.0], [0, 640.0, 240.0], [0, 0, 1]])


def _rotation(axis, degrees):
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _project(rotation):
    camera_points = (rotation @ HEAD_MODEL_POINTS.T).T + np.array([0.0, 0.0, 600.0])
    uv = (CAMERA @ camera_points.T).T
    return uv[:, :2] / uv[:, 2:3]


@pytest.mark.parametrize("pitch", [-20, -10, 0, 10, 20])
def test_head_pose_recovers_pitch_with_down_positive(pitch):
    est_pitch, est_yaw, est_roll = solve_head_pose(_project(_rotation("x", pitch)), FRAME_SHAPE)
    assert est_pitch == pytest.approx(pitch, abs=0.5)
    assert est_yaw == pytest.approx(0.0, abs=0.5)
    assert est_roll == pytest.approx(0.0, abs=0.5)


def test_head_pose_recovers_yaw_and_roll():
    assert solve_head_pose(_project(_rotation("y", 20)), FRAME_SHAPE)[1] == pytest.approx(20, abs=0.5)
    assert solve_head_pose(_project(_rotation("z", 15)), FRAME_SHAPE)[2] == pytest.approx(15, abs=0.5)


def test_head_pose_does_not_wrap_around_frontal_view():
    # The old decomposition returned values near +/-180 here and flipped sign with pixel noise,
    # which the pitch EMA then averaged into garbage.
    rng = np.random.default_rng(0)
    frontal = _project(np.eye(3))
    pitches = [solve_head_pose(frontal + rng.normal(0, 1.5, frontal.shape), FRAME_SHAPE)[0] for _ in range(50)]
    assert max(abs(p) for p in pitches) < 10


def _synthetic_person(face_conf):
    kpts = np.zeros((17, 2), np.float32)
    kpts[:5] = _project(np.eye(3))
    kpts[5], kpts[6] = (400, 330), (240, 330)
    kpts[11], kpts[12] = (380, 470), (260, 470)
    confs = np.full(17, 0.9, np.float32)
    confs[1:5] = face_conf
    return kpts, confs


def test_detection_carries_angles_only_when_face_is_visible():
    kpts, confs = _synthetic_person(face_conf=0.9)
    visible = PersonDetector._build_detection(kpts, confs, 640, 480)
    assert visible["pitch"] == pytest.approx(0.0, abs=0.5)

    kpts, confs = _synthetic_person(face_conf=KEYPOINT_CONF - 0.1)
    hidden = PersonDetector._build_detection(kpts, confs, 640, 480)
    assert hidden is not None and "pitch" not in hidden


def test_nms_keeps_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.8, 0.9, 0.7], np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [1, 2]


def test_decoder_matches_ultralytics_reference():
    ultralytics = pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    assets = Path(ultralytics.__file__).parent / "assets"
    reference = YOLO(str(MODEL), task="pose")
    detector = PersonDetector(str(MODEL), providers=["CPUExecutionProvider"])

    compared = 0
    for name in ("bus.jpg", "zidane.jpg"):
        image = cv2.imread(str(assets / name))
        for frame in (image, np.ascontiguousarray(image[:480, :640])):
            ref = reference(frame, verbose=False)[0]
            padded, gain, pad = letterbox(frame, detector.input_size)
            blob = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
            output = detector.session.run(None, {detector.input_name: blob})[0]
            boxes, scores, kpts = decode_pose_output(output, gain, pad, frame.shape)

            ref_kpts = ref.keypoints.data.cpu().numpy()
            assert len(kpts) == len(ref_kpts), name
            np.testing.assert_allclose(boxes, ref.boxes.xyxy.cpu().numpy(), atol=1.0)
            np.testing.assert_allclose(scores, ref.boxes.conf.cpu().numpy(), atol=1e-3)
            np.testing.assert_allclose(kpts[..., :2], ref_kpts[..., :2], atol=1.0)
            np.testing.assert_allclose(kpts[..., 2], ref_kpts[..., 2], atol=1e-3)
            compared += len(kpts)

    assert compared > 0
    people = detector.detect(cv2.imread(str(assets / "zidane.jpg")))
    assert people and all("pose" in p and len(p["box"]) == 4 for p in people)
