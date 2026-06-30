import cv2
import numpy as np
import math
import os
import onnxruntime as ort

# Monkey-patch ONNX Runtime to force QNN Execution Provider if Hexagon NPU is present
original_InferenceSession = ort.InferenceSession

class QNNInferenceSession(original_InferenceSession):
    def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
        if os.path.exists('/usr/lib/libQnnHtp.so'):
            providers = ['QNNExecutionProvider', 'CPUExecutionProvider']
            provider_options = [
                {
                    'backend_path': '/usr/lib/libQnnHtp.so',
                    'htp_performance_mode': 'burst',
                    'enable_htp_fp16_precision': '1'
                },
                {}
            ]
        else:
            providers = ['CPUExecutionProvider']
            provider_options = [{}]
        super().__init__(path_or_bytes, sess_options, providers, provider_options, **kwargs)

ort.InferenceSession = QNNInferenceSession

try:
    from ultralytics import YOLO
    HAVE_YOLO = True
except ImportError:
    HAVE_YOLO = False
    print("[!] Ultralytics YOLO not found.")

class AIInferenceEngine:
    def __init__(self):
        if HAVE_YOLO:
            if os.path.exists('/usr/lib/libQnnHtp.so') and os.path.exists('yolov8n-pose.onnx'):
                print("[*] Loading YOLOv8 ONNX natively through patched QNNExecutionProvider...")
                self.pose_model = YOLO('yolov8n-pose.onnx', task='pose')
            else:
                self.pose_model = YOLO('yolov8n-pose.pt')
        else:
            self.pose_model = None
            
        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self.last_roll = 0.0

    def compute_3d_head_pose(self, rmat):
        pitch = np.arctan2(-rmat[2, 1], rmat[2, 2]) * 180.0 / np.pi
        yaw = np.arctan2(rmat[2, 0], np.sqrt(rmat[2, 1]**2 + rmat[2, 2]**2)) * 180.0 / np.pi
        roll = np.arctan2(rmat[1, 0], rmat[0, 0]) * 180.0 / np.pi
        
        if np.isnan(pitch) or np.isinf(pitch) or np.isnan(yaw) or np.isinf(yaw) or np.isnan(roll) or np.isinf(roll):
            return self.last_pitch, self.last_yaw, self.last_roll
            
        pitch = float(np.clip(pitch, -45.0, 45.0))
        yaw = float(np.clip(yaw, -45.0, 45.0))
        roll = float(np.clip(roll, -45.0, 45.0))
        
        self.last_pitch, self.last_yaw, self.last_roll = pitch, yaw, roll
        return pitch, yaw, roll

    def run_inference(self, frame):
        h, w = frame.shape[:2]
        detections = []
        
        if self.pose_model:
            results = self.pose_model(frame, verbose=False)
            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                keypoints = r.keypoints.xy.cpu().numpy()
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    kpts = keypoints[i]
                    
                    x1, y1, x2, y2 = box
                    bw = x2 - x1
                    bh = y2 - y1
                    
                    if bw <= 0 or bh <= 0 or x1 < 0 or y1 < 0:
                        continue
                        
                    box_area = bw * bh
                    frame_area = w * h
                    
                    if len(kpts) < 13:
                        continue
                        
                    nose = kpts[0]
                    l_eye = kpts[1]
                    r_eye = kpts[2]
                    l_ear = kpts[3]
                    r_ear = kpts[4]
                    l_shoulder = kpts[5]
                    r_shoulder = kpts[6]
                    l_hip = kpts[11]
                    r_hip = kpts[12]
                    
                    # Absolute shoulder width in normalized pixel units
                    shoulder_width = np.abs(l_shoulder[0] - r_shoulder[0]) / w
                    
                    # Focal target engagement filters:
                    # - If shoulder width < 0.18 (background noise / too far)
                    # - If bounding box area covers > 85% of frame area (edge explosion)
                    if shoulder_width < 0.18 or box_area > (0.85 * frame_area):
                        continue
                        
                    landmarks_5 = np.array([nose, l_eye, r_eye, l_ear, r_ear], dtype=np.float32)
                    shoulders = [l_shoulder, r_shoulder]
                    hips = [l_hip, r_hip]
                    
                    pitch, yaw, roll = self._solve_pnp(landmarks_5, (h, w))
                    
                    roi = frame[int(y1):int(y2), int(x1):int(x2)]
                    if roi.size > 0:
                        roi_resized = cv2.resize(roi, (128, 128))
                        embedding = np.mean(roi_resized, axis=(0, 1)).astype(np.float32)
                        embedding = np.tile(embedding, 43)[:128]
                        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
                    else:
                        embedding = np.zeros(128, dtype=np.float32)
                        
                    detections.append({
                        "box": box,
                        "landmarks": landmarks_5,
                        "nose": nose,
                        "l_eye": l_eye,
                        "r_eye": r_eye,
                        "shoulders": shoulders,
                        "hips": hips,
                        "embedding": embedding,
                        "pitch": pitch,
                        "yaw": yaw,
                        "roll": roll
                    })
        return detections

    def _solve_pnp(self, landmarks, frame_shape):
        model_points = np.array([
            [0.0, 0.0, 0.0],
            [-30.0, -35.0, -35.0],
            [30.0, -35.0, -35.0],
            [-70.0, -10.0, -110.0],
            [70.0, -10.0, -110.0]
        ], dtype=np.float64)
        
        image_points = np.array(landmarks, dtype=np.float64)
        h, w = frame_shape[:2]
        focal_length = 640.0
        center = (w/2, h/2)
        cam_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4,1))
        rvec_guess = np.zeros((3, 1), dtype=np.float64)
        tvec_guess = np.array([[0.0], [0.0], [focal_length]], dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            model_points, image_points, cam_matrix, dist_coeffs, 
            rvec_guess, tvec_guess, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success: 
            return self.last_pitch, self.last_yaw, self.last_roll
            
        rmat, _ = cv2.Rodrigues(rvec)
        return self.compute_3d_head_pose(rmat)

    def extract_pupil_gaze(self, frame, l_eye, r_eye):
        try:
            ex, ey = int(l_eye[0]), int(l_eye[1])
            eye_dist = np.linalg.norm(np.array(l_eye) - np.array(r_eye))
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
