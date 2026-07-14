import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from core.tracking.geometry import Keypoint

logger = logging.getLogger(__name__)


class PersonDetector:
    """
    Executes Person Detection (YOLOv8 Pose).
    This implementation wraps the ultralytics native implementation
    for 100% backward compatibility during the Edge AI migration,
    but handles hardware acceleration fallbacks automatically.
    """

    def __init__(self, model_dir: str = None):

        try:
            from ultralytics import YOLO
            from core.model_manager import ModelManager

            yolo_path = ModelManager.get_model_path("yolo_pose")
            
            if yolo_path:
                self.stage1_model = YOLO(yolo_path, task="pose")
                logger.info(f"PersonDetector Initialized: {yolo_path}")
            else:
                self.stage1_model = None
                logger.warning("YOLO Pose model missing. Detection disabled.")
        except Exception as e:
            logger.error(f"PersonDetector YOLO initialization failed: {e}")
            self.stage1_model = None

        self.last_pitch, self.last_yaw, self.last_roll = 0.0, 0.0, 0.0

    def _solve_pnp(
        self, landmarks_6: np.ndarray, frame_shape: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """PnP Head Pose estimation stub, transferred from old pipeline."""
        import cv2
        import math

        h, w = frame_shape
        focal_length = w
        camera_matrix = np.array(
            [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]],
            dtype="double",
        )
        dist_coeffs = np.zeros((4, 1))

        model_points = np.array(
            [
                (0.0, 0.0, 0.0),  # Nose tip
                (-30.0, -30.0, -30.0),  # Left eye
                (30.0, -30.0, -30.0),  # Right eye
                (-50.0, 0.0, -50.0),  # Left ear
                (50.0, 0.0, -50.0),  # Right ear
                (0.0, 50.0, -20.0),  # Neck center
            ],
            dtype=np.float32,
        )

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            landmarks_6,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        pitch, yaw, roll = [
            math.degrees(math.radians(angle[0])) for angle in euler_angles
        ]

        self.last_pitch, self.last_yaw, self.last_roll = pitch, yaw, roll
        return pitch, yaw, roll

    def extract_pupil_gaze(
        self, frame: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray
    ) -> Tuple[float, bool]:
        import cv2

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
                    is_looking_away = abs(gaze_x) > 0.75
                    return gaze_x, is_looking_away
            return 0.0, False
        except Exception:
            return 0.0, False

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        h, w = frame.shape[:2]
        detections = []
        if not self.stage1_model:
            return detections
        try:
            results = self.stage1_model(frame, verbose=False)
        except Exception:
            return detections

        for r in results:
            if r.keypoints is None or r.keypoints.xy is None:
                continue
            keypoints = r.keypoints.xy.cpu().numpy()
            keypoints_conf = (
                r.keypoints.conf.cpu().numpy()
                if (hasattr(r.keypoints, "conf") and r.keypoints.conf is not None)
                else None
            )

            for i in range(len(keypoints)):
                kpts = keypoints[i]
                confs = (
                    keypoints_conf[i]
                    if keypoints_conf is not None
                    else np.ones(len(kpts))
                )
                if (
                    len(kpts) < 7
                    or confs[0] <= 0.40
                    or confs[5] <= 0.40
                    or confs[6] <= 0.40
                ):
                    continue
                valid_kpts = [kpts[j] for j in range(len(kpts)) if confs[j] > 0.40]
                if not valid_kpts:
                    continue
                xs = [kp[0] for kp in valid_kpts]
                ys = [kp[1] for kp in valid_kpts]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bw, bh = x_max - x_min, y_max - y_min
                if bw < 0.18 * w or (bw * bh) > (0.85 * w * h):
                    continue
                if len(kpts) < 13:
                    continue
                nose, l_eye, r_eye = kpts[0], kpts[1], kpts[2]
                l_ear, r_ear = kpts[3], kpts[4]
                l_shoulder, r_shoulder = kpts[5], kpts[6]
                l_hip, r_hip = kpts[11], kpts[12]
                pose = {
                    "nose": Keypoint(nose[0], nose[1], w, h),
                    "left_ear": Keypoint(l_ear[0], l_ear[1], w, h),
                    "right_ear": Keypoint(r_ear[0], r_ear[1], w, h),
                    "left_shoulder": Keypoint(l_shoulder[0], l_shoulder[1], w, h),
                    "right_shoulder": Keypoint(r_shoulder[0], r_shoulder[1], w, h),
                    "left_hip": Keypoint(l_hip[0], l_hip[1], w, h),
                    "right_hip": Keypoint(r_hip[0], r_hip[1], w, h),
                }
                has_valid_face = (
                    confs[1] > 0.40
                    and confs[2] > 0.40
                    and confs[3] > 0.40
                    and confs[4] > 0.40
                )
                neck = [
                    (l_shoulder[0] + r_shoulder[0]) / 2.0,
                    (l_shoulder[1] + r_shoulder[1]) / 2.0,
                ]
                landmarks_6 = np.array(
                    [nose, l_eye, r_eye, l_ear, r_ear, neck], dtype=np.float32
                )
                if has_valid_face:
                    pitch, yaw, roll = self._solve_pnp(landmarks_6, (h, w))
                else:
                    pitch, yaw, roll = self.last_pitch, self.last_yaw, self.last_roll

                detections.append(
                    {
                        "box": np.array([x_min, y_min, x_max, y_max]),
                        "landmarks": landmarks_6,
                        "pose": pose,
                        "nose": nose,
                        "l_eye": l_eye,
                        "r_eye": r_eye,
                        "shoulders": [l_shoulder, r_shoulder],
                        "hips": [l_hip, r_hip],
                        "pitch": pitch,
                        "yaw": yaw,
                        "roll": roll,
                        "roi_frame": frame[
                            int(y_min) : int(y_max), int(x_min) : int(x_max)
                        ],
                    }
                )
        return detections
