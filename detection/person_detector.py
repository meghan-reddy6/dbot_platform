import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.tracking.geometry import Keypoint

logger = logging.getLogger(__name__)

# Ultralytics predict() defaults, kept so detections match the previous Ultralytics-based path.
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
MAX_DETECTIONS = 300
PAD_VALUE = 114
KEYPOINT_CONF = 0.40

# Generic head (mm) for a subject facing the camera, in camera axes: x = image right (the
# subject's left), y = down, z = away from the camera. Row order = COCO keypoints 0..4.
HEAD_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # nose
        (30.0, -30.0, 30.0),  # left eye
        (-30.0, -30.0, 30.0),  # right eye
        (50.0, 0.0, 50.0),  # left ear
        (-50.0, 0.0, 50.0),  # right ear
    ],
    dtype=np.float64,
)


def letterbox(image: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize keeping aspect ratio and pad to size x size (same arithmetic as Ultralytics LetterBox)."""
    h, w = image.shape[:2]
    gain = min(size / h, size / w)
    new_w, new_h = round(w * gain), round(h * gain)
    if (new_w, new_h) != (w, h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    padded = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    return padded, gain, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS over xyxy boxes; returns kept indices, highest score first."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        inter = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None) * np.clip(
            np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None
        )
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def decode_pose_output(
    output: np.ndarray,
    gain: float,
    pad: Tuple[int, int],
    frame_shape: Tuple[int, int],
    conf_threshold: float = CONF_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
    max_detections: int = MAX_DETECTIONS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a YOLOv8-pose (1, 5 + 3K, N) output into frame-space boxes, scores and (K, 3) keypoints."""
    preds = output[0].T
    preds = preds[preds[:, 4] > conf_threshold]
    if not len(preds):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 17, 3), np.float32)

    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    keep = nms(boxes, preds[:, 4], iou_threshold)[:max_detections]
    boxes, scores = boxes[keep], preds[keep, 4]
    keypoints = preds[keep, 5:].reshape(len(keep), -1, 3).copy()

    h, w = frame_shape[:2]
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad[0]) / gain, 0, w)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad[1]) / gain, 0, h)
    keypoints[..., 0] = np.clip((keypoints[..., 0] - pad[0]) / gain, 0, w)
    keypoints[..., 1] = np.clip((keypoints[..., 1] - pad[1]) / gain, 0, h)
    return boxes, scores, keypoints


def solve_head_pose(
    face_points: np.ndarray, frame_shape: Tuple[int, int]
) -> Optional[Tuple[float, float, float]]:
    """(pitch, yaw, roll) in degrees from the nose, eyes and ears; pitch > 0 means head tilted down."""
    h, w = frame_shape[:2]
    camera_matrix = np.array([[w, 0, w / 2.0], [0, w, h / 2.0], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        HEAD_MODEL_POINTS,
        np.asarray(face_points, dtype=np.float64).reshape(-1, 1, 2),
        camera_matrix,
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(rvec)
    pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -rotation[2, 0]))))
    roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return pitch, yaw, roll


class PersonDetector:
    """YOLOv8-pose person detection on ONNX Runtime."""

    def __init__(self, model_path: Optional[str] = None, providers: Optional[List[str]] = None):
        from core.model_manager import ModelManager

        self.session = None
        self.input_size = 640
        path = model_path or ModelManager.get_model_path("yolo_pose")
        if not path or not str(path).endswith(".onnx"):
            logger.warning(f"YOLO pose ONNX model missing ({path}). Detection disabled.")
            return

        if providers is None:
            from utils.hardware import HardwareDetector

            providers = HardwareDetector.detect().get_ort_providers()
        self.session = self._create_session(str(path), providers)
        if self.session is None:
            return

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        if isinstance(model_input.shape[2], int):
            self.input_size = model_input.shape[2]
        logger.info(f"PersonDetector initialized: {path} on {self.session.get_providers()}")

    @staticmethod
    def _create_session(path: str, providers: List[str]):
        from utils.hardware import create_session

        attempts = [providers] if providers == ["CPUExecutionProvider"] else [providers, ["CPUExecutionProvider"]]
        for attempt in attempts:
            try:
                session = create_session(path, attempt)
                # Some accelerators accept the graph but fail on the first run; find out now, not per frame.
                model_input = session.get_inputs()[0]
                size = model_input.shape[2] if isinstance(model_input.shape[2], int) else 640
                session.run(None, {model_input.name: np.zeros((1, 3, size, size), np.float32)})
                return session
            except Exception as e:
                logger.error(f"PersonDetector could not start on {attempt}: {e}")
        return None

    @property
    def providers(self) -> List[str]:
        return self.session.get_providers() if self.session else []

    def extract_pupil_gaze(
        self,
        frame: np.ndarray,
        left_eye: np.ndarray,
        right_eye: np.ndarray,
        threshold: float = 0.75,
    ) -> Tuple[float, bool]:
        try:
            ex, ey = int(left_eye[0]), int(left_eye[1])
            eye_dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
            roi_size = max(10, int(eye_dist * 0.25))
            h, w = frame.shape[:2]
            y1, y2 = max(0, ey - roi_size), min(h, ey + roi_size)
            x1, x2 = max(0, ex - roi_size), min(w, ex + roi_size)

            eye_roi = frame[y1:y2, x1:x2]
            if eye_roi.size == 0:
                return 0.0, False

            gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    eye_width = roi_size * 2
                    eye_center_x = roi_size
                    gaze_x = (cx - eye_center_x) / (eye_width / 2.0)
                    is_looking_away = abs(gaze_x) > threshold
                    return gaze_x, is_looking_away
            return 0.0, False
        except Exception:
            return 0.0, False

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self.session is None:
            return []
        h, w = frame.shape[:2]
        padded, gain, pad = letterbox(frame, self.input_size)
        blob = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        output = self.session.run(None, {self.input_name: blob})[0]
        _, _, keypoints = decode_pose_output(output, gain, pad, (h, w))

        detections = []
        for person_kpts in keypoints:
            detection = self._build_detection(person_kpts[:, :2], person_kpts[:, 2], w, h)
            if detection is not None:
                detections.append(detection)
        return detections

    @staticmethod
    def _build_detection(kpts: np.ndarray, confs: np.ndarray, w: int, h: int) -> Optional[Dict[str, Any]]:
        if len(kpts) < 13 or min(confs[0], confs[5], confs[6]) <= KEYPOINT_CONF:
            return None
        valid = kpts[confs > KEYPOINT_CONF]
        x_min, y_min = valid.min(axis=0)
        x_max, y_max = valid.max(axis=0)
        bw, bh = x_max - x_min, y_max - y_min
        if bw < 0.18 * w or (bw * bh) > (0.85 * w * h):
            return None

        nose, l_eye, r_eye = kpts[0], kpts[1], kpts[2]
        l_ear, r_ear = kpts[3], kpts[4]
        l_shoulder, r_shoulder = kpts[5], kpts[6]
        l_hip, r_hip = kpts[11], kpts[12]
        detection = {
            "box": np.array([x_min, y_min, x_max, y_max]),
            "pose": {
                "nose": Keypoint(nose[0], nose[1], w, h),
                "left_ear": Keypoint(l_ear[0], l_ear[1], w, h),
                "right_ear": Keypoint(r_ear[0], r_ear[1], w, h),
                "left_shoulder": Keypoint(l_shoulder[0], l_shoulder[1], w, h),
                "right_shoulder": Keypoint(r_shoulder[0], r_shoulder[1], w, h),
                "left_hip": Keypoint(l_hip[0], l_hip[1], w, h),
                "right_hip": Keypoint(r_hip[0], r_hip[1], w, h),
            },
            "nose": nose,
            "l_eye": l_eye,
            "r_eye": r_eye,
        }
        # Head angles only when the whole face is visible; callers keep the track's previous angles otherwise.
        if min(confs[1:5]) > KEYPOINT_CONF:
            angles = solve_head_pose(kpts[:5], (h, w))
            if angles is not None:
                detection["pitch"], detection["yaw"], detection["roll"] = angles
        return detection
