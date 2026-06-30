import time
import logging
import threading
import numpy as np
from collections import deque, Counter

logger = logging.getLogger("DeskBotV3.Tracking")
state_mutex = threading.Lock()

try:
    import pyttsx3
    HAVE_PYTTSX3 = True
except ImportError:
    HAVE_PYTTSX3 = False
    print("[!] pyttsx3 not installed. Voice alerts will be muted.")

class Target:
    def __init__(self, track_id, embedding, box):
        self.track_id = track_id
        self.embedding = embedding
        self.box = box
        self.name = "Unknown"
        self.state = "Unregistered Guest"
        self.last_state = "Unregistered Guest"
        self.is_verified = False
        
        self.last_seen = time.time()
        self.last_update = time.time()
        
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.gaze_x = 0.0
        
        self.is_standing = False
        self.is_looking_away = False
        
        self.sitting_duration_clock = 0.0
        self.standing_duration_clock = 0.0
        self.gaze_away_clock = 0.0
        self.slouch_timer = 0.0
        
        self.slouch_sensitivity = 15.0
        self.session_limit = 1200
        self.stand_requirement = 180
        self.gaze_away_limit = 20
        self.biometric_cutoff = 0.55
        
        self.baseline_torso_ratio = 0.0     
        self.baseline_pitch = 0.0           
        self.baseline_shoulder_y = 0.0      
        self.calibration_accumulator = []  
        self.calibration_start = None
        self.calibration_announced = False  
        self.last_log_time = 0.0            
        
        self.state_history_window = deque(maxlen=20)  
        self.recovery_calibration_start = None
        self.recovery_accumulator = []
        
        self.smoothed_pitch = None
        self.smoothed_ratio = None
        self.smoothed_y = None
        self.last_y = 0.0

    def get_centroid(self):
        return ((self.box[0] + self.box[2])/2, (self.box[1] + self.box[3])/2)

class TrackerEngine:
    def __init__(self, inference_engine, db_manager):
        self.targets = {}
        self.track_counter = 0
        self.db_manager = db_manager
        self.profiles = self.db_manager.load_all_profiles()
        self.mutex = threading.Lock()
        self.inference_engine = inference_engine
        self.last_voice_alert = 0.0

    def sync_profiles(self):
        with self.mutex:
            self.profiles = self.db_manager.load_all_profiles()
            for target in self.targets.values():
                if target.name in self.profiles:
                    p = self.profiles[target.name]
                    target.slouch_sensitivity = p["slouch_sensitivity"]
                    target.session_limit = p["session_limit"]
                    target.stand_requirement = p["stand_requirement"]
                    target.gaze_away_limit = p["gaze_away_limit"]
                    target.biometric_cutoff = p.get("biometric_cutoff", 0.55)

    def _match_profile(self, embedding):
        best_match = None
        best_dist = float('inf')
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-6)
        for name, profile in self.profiles.items():
            db_emb = profile["embedding"]
            db_emb_norm = db_emb / (np.linalg.norm(db_emb) + 1e-6)
            cosine_similarity = np.dot(emb_norm, db_emb_norm)
            dist = 1.0 - cosine_similarity
            cutoff = profile.get("biometric_cutoff", 0.55)
            if dist <= cutoff and dist < best_dist:
                best_dist = dist
                best_match = name
        return best_match

    def _dispatch_voice(self, text):
        now = time.time()
        if now - self.last_voice_alert > 25.0:
            self.last_voice_alert = now
            def voice_worker():
                if not HAVE_PYTTSX3: return
                import pyttsx3
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 145)
                    engine.say(text)
                    engine.runAndWait()
                    engine.proxy.disconnect()
                    del engine
                except Exception: pass
            threading.Thread(target=voice_worker, daemon=True).start()

    def _evaluate_single_target_health(self, target, pose, current_ratio, dt, current_time):
        if target.recovery_calibration_start is not None:
            elapsed_recovery = current_time - target.recovery_calibration_start
            shoulder_width = np.abs(pose['right_shoulder'].x - pose['left_shoulder'].x)
            shoulder_center_y = (pose['left_shoulder'].y + pose['right_shoulder'].y) / 2.0
            current_ratio_computed = np.abs(shoulder_center_y - pose['nose'].y) / max(shoulder_width, 1e-6)
            target.recovery_accumulator.append((current_ratio_computed, shoulder_center_y))
            
            if elapsed_recovery >= 1.0:
                if target.recovery_accumulator:
                    target.baseline_torso_ratio = float(np.mean([i[0] for i in target.recovery_accumulator]))
                    target.baseline_shoulder_y = float(np.mean([i[1] for i in target.recovery_accumulator]))
                target.recovery_calibration_start = None
                target.recovery_accumulator = []
            return

        if target.state == "Calibrating":
            if target.calibration_start is None:
                target.calibration_start = current_time
                target.calibration_accumulator = []
            if not getattr(target, 'calibration_announced', False):
                target.calibration_announced = True
                self._dispatch_voice("Please look straight ahead in a comfortable posture to calibrate your desk setup.")
                
            elapsed_calib = current_time - target.calibration_start
            shoulder_width = np.abs(pose['right_shoulder'].x - pose['left_shoulder'].x)
            shoulder_center_y = (pose['left_shoulder'].y + pose['right_shoulder'].y) / 2.0
            current_ratio_computed = np.abs(shoulder_center_y - pose['nose'].y) / max(shoulder_width, 1e-6)
            
            if current_time - getattr(target, 'last_log_time', 0.0) >= 1.0:
                target.last_log_time = current_time
                remaining = max(0.0, 3.0 - elapsed_calib)
                print(f"[!] PROMPT: {target.name}, calibrating... {remaining:.1f}s remaining.")
                
            target.calibration_accumulator.append((target.pitch, current_ratio_computed, shoulder_center_y))
            
            if elapsed_calib >= 3.0:
                if target.calibration_accumulator:
                    target.baseline_pitch = float(np.mean([i[0] for i in target.calibration_accumulator]))
                    target.baseline_torso_ratio = float(np.mean([i[1] for i in target.calibration_accumulator]))
                    target.baseline_shoulder_y = float(np.mean([i[2] for i in target.calibration_accumulator]))
                target.calibration_accumulator = []
                target.state = "Tracking Active"
                self._dispatch_voice("Calibration successful. Posture monitoring is now active.")
            return

        if target.state in ["Tracking Active", "Standing", "Looking Away", "Ocular Break Recommended", "Session Limit Reached - Stand Up!", "Posture Deficit Alert"]:
            shoulder_width = np.abs(pose['right_shoulder'].x - pose['left_shoulder'].x)
            shoulder_center_y = (pose['left_shoulder'].y + pose['right_shoulder'].y) / 2.0
            
            pitch_val = target.smoothed_pitch if target.smoothed_pitch is not None else target.pitch
            ratio_val = target.smoothed_ratio if target.smoothed_ratio is not None else current_ratio

            relative_slouch = pitch_val - target.baseline_pitch
            is_slouching = (relative_slouch > target.slouch_sensitivity) or (ratio_val < (target.baseline_torso_ratio * 0.88))
            
            normalized_height_delta = (target.baseline_shoulder_y - shoulder_center_y) / max(shoulder_width, 1e-6)
            is_standing = (normalized_height_delta > 0.45) or (ratio_val > (target.baseline_torso_ratio * 1.35))
            
            target.is_standing = is_standing

            if is_standing: 
                frame_candidate = "Standing"
            elif is_slouching: 
                frame_candidate = "Posture Deficit Alert"
            else: 
                frame_candidate = "Tracking Active"

            target.state_history_window.append(frame_candidate)

            if len(target.state_history_window) >= 10:
                target.state = Counter(target.state_history_window).most_common(1)[0][0]
            else:
                target.state = frame_candidate

            if not is_standing and target.state == "Standing":
                target.state = "Tracking Active"
                target.state_history_window.clear()
                target.baseline_shoulder_y = (target.baseline_shoulder_y * 0.7) + (shoulder_center_y * 0.3)
                target.baseline_torso_ratio = (target.baseline_torso_ratio * 0.7) + (current_ratio * 0.3)

            if target.is_looking_away and target.state == "Standing" and abs(target.gaze_x) <= 0.60:
                target.is_looking_away = False

            if target.is_looking_away:
                target.gaze_away_clock += dt
                target.state = "Ocular Break Recommended" if target.gaze_away_clock > target.gaze_away_limit else "Looking Away"
                return
            else:
                target.gaze_away_clock = 0.0

            if target.state == "Standing":
                target.standing_duration_clock += dt
                target.slouch_timer = max(0.0, target.slouch_timer - dt)
                if target.standing_duration_clock >= target.stand_requirement:
                    target.sitting_duration_clock = 0.0
                if is_standing:
                    target.baseline_shoulder_y = (target.baseline_shoulder_y * 0.95) + (shoulder_center_y * 0.05)
            else:
                target.sitting_duration_clock += dt
                if target.sitting_duration_clock >= target.session_limit:
                    target.state = "Session Limit Reached - Stand Up!"
                    target.standing_duration_clock = 0.0
                else:
                    if target.state == "Posture Deficit Alert":
                        target.slouch_timer += dt
                        if target.slouch_timer >= 8.0:
                            self._dispatch_voice(f"Please correct your posture, {target.name}.")
                    else:
                        target.slouch_timer = max(0.0, target.slouch_timer - dt)

    def update(self, frame, detections, frame_shape):
        return self.process_frame_mot(frame, detections, frame_shape)

    def process_frame_mot(self, frame, detections, frame_shape):
        current_time = time.time()
        h, w = frame_shape[:2]
        active_ids = set()
        
        assigned_names = set()
        
        ranked_detections = []
        for det in detections:
            box = det["box"]
            centroid_x = (box[0] + box[2]) / 2.0
            shoulder_width = np.abs(det["pose"]["left_shoulder"].x - det["pose"]["right_shoulder"].x)
            score = shoulder_width - (np.abs(centroid_x - w/2) / w)
            ranked_detections.append((score, det))
        ranked_detections.sort(key=lambda x: x[0], reverse=True)
        
        with self.mutex:
            primary_det = ranked_detections[0][1] if ranked_detections else None
            
            for score, det in ranked_detections:
                embedding = det["embedding"]
                box = det["box"]
                pose = det["pose"]
                det_centroid = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                shoulder_width = np.abs(pose["left_shoulder"].x - pose["right_shoulder"].x)
                
                is_primary = (det is primary_det)
                matched_target = None
                best_cost = float('inf')
                
                for tid, target in self.targets.items():
                    tc = target.get_centroid()
                    
                    pixel_distance = np.linalg.norm(np.array(det_centroid) - np.array(tc))
                    if pixel_distance > (w * 0.25):
                        continue
                        
                    scale_aware_dist = pixel_distance / (shoulder_width * w + 1e-6)
                    cost = (scale_aware_dist * 0.5) + (np.linalg.norm(embedding - target.embedding) * 0.5)
                    
                    max_cost_limit = 0.70 if target.state == "Coasting (Paused)" else 0.55
                    if len(set(target.state_history_window)) > 1:
                        max_cost_limit = max(max_cost_limit, 0.85)
                    
                    if cost <= max_cost_limit and cost < best_cost:
                        best_cost = cost; matched_target = target
                        
                if matched_target is None:
                    profile_name = self._match_profile(embedding)
                    matched_coasting_target = None
                    if profile_name and profile_name not in assigned_names:
                        for tid, target in self.targets.items():
                            if target.name == profile_name and target.state == "Coasting (Paused)":
                                matched_coasting_target = target; break
                                
                    if matched_coasting_target is not None:
                        matched_target = matched_coasting_target
                        matched_target.state = "Tracking Active"
                        matched_target.last_seen = current_time
                        matched_target.last_update = current_time
                        matched_target.recovery_calibration_start = current_time
                        matched_target.recovery_accumulator = []
                    else:
                        track_hash = f"Person_hash_{int(current_time * 1000)}_{self.track_counter}"
                        self.track_counter += 1
                        matched_target = Target(track_hash, embedding, box)
                        self.targets[track_hash] = matched_target

                dt = current_time - matched_target.last_update
                if matched_target.state == "Coasting (Paused)":
                    matched_target.state = "Tracking Active"
                    matched_target.recovery_calibration_start = current_time
                    matched_target.recovery_accumulator = []

                matched_target.last_seen = current_time
                matched_target.last_update = current_time
                matched_target.box = box
                active_ids.add(matched_target.track_id)
                
                matched_target.pitch = (matched_target.pitch * 0.6) + (det["pitch"] * 0.4)
                matched_target.yaw = det["yaw"]
                matched_target.roll = det["roll"]
                
                shoulder_center_y = (pose["left_shoulder"].y + pose["right_shoulder"].y) / 2.0
                current_ratio = np.abs(shoulder_center_y - pose["nose"].y) / max(shoulder_width, 1e-6)
                matched_target.last_y = shoulder_center_y
                
                if matched_target.smoothed_pitch is None:
                    matched_target.smoothed_pitch = matched_target.pitch
                    matched_target.smoothed_ratio = current_ratio
                    matched_target.smoothed_y = shoulder_center_y
                else:
                    alpha = 0.15
                    matched_target.smoothed_pitch = (1 - alpha) * matched_target.smoothed_pitch + alpha * matched_target.pitch
                    matched_target.smoothed_ratio = (1 - alpha) * matched_target.smoothed_ratio + alpha * current_ratio
                    matched_target.smoothed_y = (1 - alpha) * matched_target.smoothed_y + alpha * shoulder_center_y
                
                profile_name = self._match_profile(embedding)
                
                if profile_name:
                    if profile_name in assigned_names:
                        matched_target.name = "Unknown"
                        matched_target.state = "Secondary Bystander"
                        matched_target.calibration_start = None
                        matched_target.calibration_accumulator = []
                    else:
                        assigned_names.add(profile_name)
                        was_unknown = (matched_target.name == "Unknown")
                        matched_target.name = profile_name
                        
                        p = self.profiles[profile_name]
                        matched_target.slouch_sensitivity = p["slouch_sensitivity"]
                        matched_target.session_limit = p["session_limit"]
                        matched_target.stand_requirement = p["stand_requirement"]
                        matched_target.gaze_away_limit = p["gaze_away_limit"]
                        matched_target.biometric_cutoff = p.get("biometric_cutoff", 0.55)
                        
                        if was_unknown:
                            matched_target.state = "Calibrating"
                            matched_target.calibration_start = current_time
                            matched_target.calibration_accumulator = []
                            matched_target.calibration_announced = False
                        elif matched_target.state in ["Unregistered Guest", "Coasting (Paused)", "Secondary Bystander"]:
                            matched_target.state = "Tracking Active"
                else:
                    if matched_target.name != "Unknown" and matched_target.name in self.profiles:
                        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-6)
                        db_emb = self.profiles[matched_target.name]["embedding"]
                        db_emb_norm = db_emb / (np.linalg.norm(db_emb) + 1e-6)
                        cosine_similarity = np.dot(emb_norm, db_emb_norm)
                        cosine_dist = 1.0 - cosine_similarity
                        if cosine_dist > matched_target.biometric_cutoff:
                            matched_target.calibration_accumulator = []
                            matched_target.calibration_start = None
                            matched_target.name = "Unknown"
                            matched_target.state = "Unregistered Guest"
                
                if matched_target.name == "Unknown":
                    if not is_primary:
                        matched_target.state = "Secondary Bystander"
                    else:
                        matched_target.state = "Unregistered Guest"
                    continue
                    
                gaze_x, is_looking_away = self.inference_engine.extract_pupil_gaze(frame, det["l_eye"], det["r_eye"])
                matched_target.gaze_x = gaze_x
                matched_target.is_looking_away = is_looking_away
                
                self._evaluate_single_target_health(matched_target, pose, current_ratio, dt, current_time)
                
                if matched_target.state != matched_target.last_state:
                    print(f"EVENT LOG | [{time.strftime('%Y-%m-%d %H:%M:%S')}] | User: {matched_target.name} | Transitioned state directly from {matched_target.last_state} to {matched_target.state} Mode.")
                    matched_target.last_state = matched_target.state
                
                if current_time - matched_target.last_log_time >= 1.0:
                    matched_target.last_log_time = current_time
                    print(f"[LOG] {time.strftime('%Y-%m-%d %H:%M:%S')} | Target: {matched_target.name} | State: {matched_target.state} | Pitch: {matched_target.pitch:+.1f}° | Torso Ratio: {current_ratio:.3f} | Sitting: {int(matched_target.sitting_duration_clock)}s | Standing: {int(matched_target.standing_duration_clock)}s")
                                    
            for tid, target in list(self.targets.items()):
                if tid not in active_ids:
                    if current_time - target.last_seen > 10.0:
                        if target.name != "Unknown":
                            self.db_manager.log_session_metrics(target.name, "session_ended", target.sitting_duration_clock)
                        del self.targets[tid]
                    else:
                        if target.state not in ["Calibrating", "Secondary Bystander"]:
                            target.state = "Coasting (Paused)"
                    if target.state != target.last_state:
                        print(f"EVENT LOG | [{time.strftime('%Y-%m-%d %H:%M:%S')}] | User: {target.name} | Transitioned state directly from {target.last_state} to {target.state} Mode.")
                        target.last_state = target.state