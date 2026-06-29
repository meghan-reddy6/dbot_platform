import cv2
import numpy as np
import math
import os

try:
    from ultralytics import YOLO
    HAVE_YOLO = True
except ImportError:
    HAVE_YOLO = False
    print("[!] Ultralytics YOLO not found.")

class AIInferenceEngine:
    def __init__(self):
        if HAVE_YOLO:
            if os.path.exists('/usr/lib/libQnnHtp.so'):
                if os.path.exists('yolov8n-pose.onnx'):
                    print("[*] Loading YOLOv8 ONNX with QNNExecutionProvider...")
                    self.pose_model = YOLO('yolov8n-pose.onnx', task='pose')
                else:
                    self.pose_model = YOLO('yolov8n-pose.pt')
            else:
                self.pose_model = YOLO('yolov8n-pose.pt')
        else:
            self.pose_model = None
            
        self.last_pitch = 0.0
        self.last_yaw = 0.0
        self.last_roll = 0.0

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
                    
                    landmarks_5 = np.array([nose, l_eye, r_eye, l_ear, r_ear], dtype=np.float32)
                    shoulders = [l_shoulder, r_shoulder]
                    hips = [l_hip, r_hip]
                    
                    pitch, yaw, roll = self._solve_pnp(landmarks_5, (h, w))
                    
                    roi = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
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
        success, rvec, tvec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs, rvec_guess, tvec_guess, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        
        if not success: 
            return self.last_pitch, self.last_yaw, self.last_roll
            
        rmat, _ = cv2.Rodrigues(rvec)
        
        try:
            pitch = np.arcsin(-rmat[2, 0]) * 180.0 / np.pi
            yaw = np.arctan2(rmat[2, 1], rmat[2, 2]) * 180.0 / np.pi
            roll = np.arctan2(rmat[1, 0], rmat[0, 0]) * 180.0 / np.pi
            
            if np.isnan(pitch) or np.isinf(pitch) or np.isnan(yaw) or np.isinf(yaw):
                pitch, yaw, roll = self.last_pitch, self.last_yaw, self.last_roll
            else:
                self.last_pitch, self.last_yaw, self.last_roll = pitch, yaw, roll
        except Exception:
            pitch, yaw, roll = self.last_pitch, self.last_yaw, self.last_roll
            
        return pitch, yaw, roll

    def extract_pupil_gaze(self, frame, l_eye, r_eye):
        try:
            ex, ey = int(l_eye[0]), int(l_eye[1])
            roi_size = 20
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
                    is_looking_away = abs(gaze_x) > 0.44
                    return gaze_x, is_looking_away
                    
            return 0.0, False
        except Exception:
            return 0.0, False
