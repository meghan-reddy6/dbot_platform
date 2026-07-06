import platform
import os
import time
import numpy as np
import cv2
import math
from typing import Any, Dict, List, Tuple

# Get dynamic base directory mapping to root of the repo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Keypoint:
    """
    Represents a normalized 2D coordinate for skeletal landmarks.
    """
    def __init__(self, x_pixel: float, y_pixel: float, w: int, h: int) -> None:
        self.x: float = float(x_pixel / w)
        self.y: float = float(y_pixel / h)

class NativeFaceCascade:
    """
    Independent Stage 2: Native Biometric Facial Execution Cascade
    Executes OpenCV HAAR Face Detection -> LBP Texture Extraction
    """
    def __init__(self, model_dir: str):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def generate_signature(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Executes cascade and produces a 128-d LBP texture signature.
        Divides the face into an 8x8 grid (64 cells). Each cell generates a 2-bin LBP histogram.
        """
        if face_crop is None or face_crop.size == 0:
            return np.zeros(128, dtype=np.float32)

        # 1. Detection
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))

        if len(faces) == 0:
            return np.zeros(128, dtype=np.float32)

        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        inner_face = gray[y:y+h, x:x+w]
        
        # 2. Extract LBP 128-d
        inner_face = cv2.resize(inner_face, (64, 64))
        lbp = np.zeros_like(inner_face)
        for i in range(1, 63):
            for j in range(1, 63):
                center = inner_face[i, j]
                code = 0
                code |= (inner_face[i-1, j-1] > center) << 7
                code |= (inner_face[i-1, j] > center) << 6
                code |= (inner_face[i-1, j+1] > center) << 5
                code |= (inner_face[i, j+1] > center) << 4
                code |= (inner_face[i+1, j+1] > center) << 3
                code |= (inner_face[i+1, j] > center) << 2
                code |= (inner_face[i+1, j-1] > center) << 1
                code |= (inner_face[i, j-1] > center) << 0
                lbp[i, j] = code

        features = []
        cell_size = 8
        for i in range(8):
            for j in range(8):
                cell = lbp[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                hist = cv2.calcHist([cell], [0], None, [2], [0, 256])
                features.extend(hist.flatten())

        embedding = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class CrossPlatformInferenceManager:
    def __init__(self) -> None:
        self.os_type = platform.system()
        self.arch_type = platform.machine().lower()
        self.stage1_model = None
        self.stage2_cascade = None
        
        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self.last_roll = 0.0
        
        print(f"[*] CrossPlatformInferenceManager Boot. OS: {self.os_type} | Arch: {self.arch_type}")
        self._initialize_hardware_backends()

    def _get_optimal_providers(self) -> List[Any]:
        """
        Dynamically builds the fallback cascade of ONNX providers based on host OS.
        """
        providers = []
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
        except ImportError:
            available = []

        if self.os_type == "Linux" and "aarch64" in self.arch_type and os.path.exists("/usr/lib/libQnnHexagon.so"):
            providers.append(("QNNExecutionProvider", {
                "backend_path": "/usr/lib/libQnnHexagon.so",
                "htp_performance_mode": "burst",
                "htp_precision": "fp16"
            }))
        elif self.os_type == "Windows" and "arm64" in self.arch_type:
            providers.append(("QNNExecutionProvider", {
                "backend_path": "QnnHtp.dll",
                "htp_performance_mode": "burst",
                "htp_precision": "fp16"
            }))
        elif self.os_type == "Darwin":
            if "CoreMLExecutionProvider" in available:
                providers.append("CoreMLExecutionProvider")
        elif self.os_type == "Windows" or self.os_type == "Linux":
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            elif "DmlExecutionProvider" in available:
                providers.append("DmlExecutionProvider")

        providers.append("CPUExecutionProvider")
        return providers

    def _initialize_hardware_backends(self) -> None:
        providers = self._get_optimal_providers()
        print(f"[*] Bound Hardware Providers: {providers}")
        
        # Load Stage 1: YOLO Pose Model natively through Ultralytics (handles its own HW logic)
        try:
            from ultralytics import YOLO
            MODELS_DIR = os.path.join(BASE_DIR, 'models')
            if self.os_type == "Windows":
                yolo_path = os.path.join(MODELS_DIR, 'yolov8n-pose.pt')
                if not os.path.exists(yolo_path):
                    yolo_path = os.path.join(MODELS_DIR, 'yolov8n-pose.onnx')
            else:
                yolo_path = os.path.join(MODELS_DIR, 'yolov8n-pose.onnx')
                if not os.path.exists(yolo_path):
                    yolo_path = os.path.join(MODELS_DIR, 'yolov8n-pose.pt')
                    
            self.stage1_model = YOLO(yolo_path, task='pose')
        except Exception as e:
            print(f"[!] YOLO initialization failed: {e}")
            self.stage1_model = None

        # Load Stage 2: Biometric Cascade using unified OpenCV backend logic
        self.stage2_cascade = NativeFaceCascade(os.path.join(BASE_DIR, "models"))

    def _solve_pnp(self, landmarks_5: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[float, float, float]:
        h, w = frame_shape
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        model_points = np.array([
            (0.0, 0.0, 0.0), (-30.0, -125.0, -30.0), (30.0, -125.0, -30.0),
            (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ], dtype="double")
        
        rvec_guess = np.zeros((3, 1), dtype=np.float64)
        tvec_guess = np.array([[0.0], [0.0], [focal_length]], dtype=np.float64)
        
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, landmarks_5, camera_matrix, dist_coeffs,
            rvec_guess, tvec_guess, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return self.last_pitch, self.last_yaw, self.last_roll
        rmat, _ = cv2.Rodrigues(rotation_vector)
        pitch = np.arctan2(-rmat[2, 1], rmat[2, 2]) * 180.0 / np.pi
        yaw = np.arctan2(rmat[2, 0], np.sqrt(rmat[2, 1]**2 + rmat[2, 2]**2)) * 180.0 / np.pi
        roll = np.arctan2(rmat[1, 0], rmat[0, 0]) * 180.0 / np.pi
        if np.isnan(pitch) or np.isinf(pitch):
            return self.last_pitch, self.last_yaw, self.last_roll
        pitch = float(np.clip(pitch, -45.0, 45.0))
        yaw = float(np.clip(yaw, -45.0, 45.0))
        roll = float(np.clip(roll, -45.0, 45.0))
        self.last_pitch, self.last_yaw, self.last_roll = pitch, yaw, roll
        return pitch, yaw, roll

    def extract_pupil_gaze(self, frame: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray) -> Tuple[float, bool]:
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
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    eye_width = roi_size * 2
                    eye_center_x = roi_size
                    gaze_x = (cx - eye_center_x) / (eye_width / 2.0)
                    is_looking_away = abs(gaze_x) > 0.22
                    return gaze_x, is_looking_away
            return 0.0, False
        except Exception:
            return 0.0, False

    def execute_stage1_detector(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        h, w = frame.shape[:2]
        detections = []
        if not self.stage1_model:
            return detections
        try:
            results = self.stage1_model(frame, verbose=False)
        except Exception:
            return detections
            
        for r in results:
            keypoints = r.keypoints.xy.cpu().numpy()
            keypoints_conf = r.keypoints.conf.cpu().numpy() if (hasattr(r.keypoints, 'conf') and r.keypoints.conf is not None) else None
            
            for i in range(len(keypoints)):
                kpts = keypoints[i]
                confs = keypoints_conf[i] if keypoints_conf is not None else np.ones(len(kpts))
                if len(kpts) < 7 or confs[0] <= 0.40 or confs[5] <= 0.40 or confs[6] <= 0.40:
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
                    "left_shoulder": Keypoint(l_shoulder[0], l_shoulder[1], w, h),
                    "right_shoulder": Keypoint(r_shoulder[0], r_shoulder[1], w, h)
                }
                has_valid_face = (confs[1] > 0.40 and confs[2] > 0.40 and confs[3] > 0.40 and confs[4] > 0.40)
                landmarks_5 = np.array([nose, l_eye, r_eye, l_ear, r_ear], dtype=np.float32)
                if has_valid_face:
                    pitch, yaw, roll = self._solve_pnp(landmarks_5, (h, w))
                else:
                    pitch, yaw, roll = self.last_pitch, self.last_yaw, self.last_roll
                
                detections.append({
                    "box": np.array([x_min, y_min, x_max, y_max]),
                    "landmarks": landmarks_5,
                    "pose": pose,
                    "nose": nose,
                    "l_eye": l_eye,
                    "r_eye": r_eye,
                    "shoulders": [l_shoulder, r_shoulder],
                    "hips": [l_hip, r_hip],
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll,
                    "roi_frame": frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                })
        return detections

    def execute_stage2_biometrics(self, face_crop: np.ndarray) -> np.ndarray:
        if self.stage2_cascade:
            return self.stage2_cascade.generate_signature(face_crop)
        else:
            return np.zeros(128, dtype=np.float32)
