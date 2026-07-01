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

class Person:
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
        
        self.screen_gaze_accumulation_timer = 0.0
        self.ocular_break_timer = 0.0
        self.gaze_away_clock = 0.0
        self.slouch_timer = 0.0
        
        self.ocular_break_announced = False
        self.session_limit_announced = False
        self.slouch_announced = False
        
        self.slouch_sensitivity = 15.0
        self.session_limit = 2400
        self.stand_requirement = 120
        self.gaze_away_limit = 20.0
        self.screen_gaze_limit = 1200.0
        self.biometric_cutoff = 0.55
        
        self.baseline_torso_ratio = 0.0     
        self.baseline_pitch = 0.0           
        self.baseline_shoulder_y = 0.0      
        self.calibration_accumulator = []  
        self.calibration_start = None
        self.calibration_announced = False  
        self.last_log_time = 0.0            
        
        self.state_history_window = deque(maxlen=25)  
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
        self.tracked_persons = {}
        self.track_counter = 0
        self.db_manager = db_manager
        self.profiles = self.db_manager.load_all_profiles()
        print(f"[BOOT] Database loaded. Active profile synchronization complete.")
        self.mutex = threading.Lock()
        self.inference_engine = inference_engine
        self.last_voice_alert = 0.0
        self.system_was_manually_cleared = False

    def sync_profiles(self):
        with self.mutex:
            self.profiles = self.db_manager.load_all_profiles()
            for person in self.tracked_persons.values():
                if person.name in self.profiles:
                    p = self.profiles[person.name]
                    person.slouch_sensitivity = p["slouch_sensitivity"]
                    person.session_limit = p["session_limit"]
                    person.stand_requirement = p["stand_requirement"]
                    person.gaze_away_limit = float(p.get("ocular_break_duration", 20.0))
                    person.screen_gaze_limit = float(p.get("screen_gaze_limit", 1200.0))
                    person.biometric_cutoff = p.get("biometric_cutoff", 0.55)

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
        if now - self.last_voice_alert > 10.0:
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

    def _evaluate_single_target_health(self, person, pose, current_ratio, dt, current_time, frame_shape):
        if person.recovery_calibration_start is not None:
            elapsed_recovery = current_time - person.recovery_calibration_start
            shoulder_width = np.abs(pose['right_shoulder'].x - pose['left_shoulder'].x)
            shoulder_center_y = (pose['left_shoulder'].y + pose['right_shoulder'].y) / 2.0
            current_ratio_computed = np.abs(shoulder_center_y - pose['nose'].y) / max(shoulder_width, 1e-6)
            person.recovery_accumulator.append((current_ratio_computed, shoulder_center_y))
            
            if elapsed_recovery >= 1.0:
                if person.recovery_accumulator:
                    person.baseline_torso_ratio = float(np.mean([i[0] for i in person.recovery_accumulator]))
                    person.baseline_shoulder_y = float(np.mean([i[1] for i in person.recovery_accumulator]))
                person.recovery_calibration_start = None
                person.recovery_accumulator = []
            return

        if person.state == "Calibrating":
            if person.calibration_start is None:
                person.calibration_start = current_time
                person.calibration_accumulator = []
            if not getattr(person, 'calibration_announced', False):
                person.calibration_announced = True
                self._dispatch_voice("Please look straight ahead in a comfortable posture to calibrate your desk setup.")
                
            elapsed_calib = current_time - person.calibration_start
            shoulder_width = np.abs(pose['right_shoulder'].x - pose['left_shoulder'].x)
            shoulder_center_y = (pose['left_shoulder'].y + pose['right_shoulder'].y) / 2.0
            current_ratio_computed = np.abs(shoulder_center_y - pose['nose'].y) / max(shoulder_width, 1e-6)
            
            if current_time - getattr(person, 'last_log_time', 0.0) >= 1.0:
                person.last_log_time = current_time
                remaining = max(0.0, 3.0 - elapsed_calib)
                # print(f"[!] PROMPT: {person.name}, calibrating... {remaining:.1f}s remaining.")
                
            person.calibration_accumulator.append((person.pitch, current_ratio_computed, shoulder_center_y))
            
            if elapsed_calib >= 3.0:
                if person.calibration_accumulator:
                    person.baseline_pitch = float(np.mean([i[0] for i in person.calibration_accumulator]))
                    person.baseline_torso_ratio = float(np.mean([i[1] for i in person.calibration_accumulator]))
                    person.baseline_shoulder_y = float(np.mean([i[2] for i in person.calibration_accumulator]))
                person.calibration_accumulator = []
                person.state = "Tracking Active"
                self._dispatch_voice("Calibration successful. Posture monitoring is now active.")
            return

        if person.state in ["Tracking Active", "Standing", "Looking Away", "Ocular Break Recommended", "Session Limit Reached - Stand Up!", "Posture Deficit Alert"]:
            
            if not person.is_looking_away:
                person.screen_gaze_accumulation_timer += dt
                if person.state != "Ocular Break Recommended":
                    person.ocular_break_timer = 0.0
            
            shoulder_width = np.abs(pose['right_shoulder'].x - pose['left_shoulder'].x)
            shoulder_center_y = (pose['left_shoulder'].y + pose['right_shoulder'].y) / 2.0
            
            pitch_val = person.smoothed_pitch if person.smoothed_pitch is not None else person.pitch
            ratio_val = person.smoothed_ratio if person.smoothed_ratio is not None else current_ratio

            relative_slouch = pitch_val - person.baseline_pitch
            is_slouching = (relative_slouch > person.slouch_sensitivity) or (ratio_val < (person.baseline_torso_ratio * 0.83))
            
            normalized_height_delta = (person.baseline_shoulder_y - shoulder_center_y) / max(shoulder_width, 1e-6)
            is_standing = (normalized_height_delta > 0.48) or (ratio_val > (person.baseline_torso_ratio * 1.40))
            
            is_pinned_to_ceiling = (person.box[1] <= frame_shape[0] * 0.05)
            nose_missing = (pose['nose'].y <= 0.01)
            
            person.is_standing = is_standing

            if is_pinned_to_ceiling and nose_missing:
                frame_candidate = "Standing"
            elif is_standing: 
                frame_candidate = "Standing"
            elif is_slouching: 
                frame_candidate = "Posture Deficit Alert"
            else: 
                frame_candidate = "Tracking Active"

            person.state_history_window.append(frame_candidate)

            if len(person.state_history_window) >= 10:
                person.state = Counter(person.state_history_window).most_common(1)[0][0]
            else:
                person.state = frame_candidate
                
            # Bi-Directional Calibration Gate (Fix Sticky False Alerts)
            if person.state == "Posture Deficit Alert":
                if current_ratio >= person.baseline_torso_ratio * 0.96 and relative_slouch <= person.slouch_sensitivity:
                    person.state_history_window.clear()
                    person.state = "Tracking Active"
                    person.fast_relatch_frames = 5

            # Wellness Timer Overrides
            if person.screen_gaze_accumulation_timer >= person.screen_gaze_limit:
                person.state = "Ocular Break Recommended"
                if not person.ocular_break_announced:
                    print(f"[TIMER ALERT] Triggering Voice Alert for: {person.state}")
                    self._dispatch_voice("Attention, eye strain warning. Please look away from the screen.")
                    person.ocular_break_announced = True
                
                if person.is_looking_away:
                    person.ocular_break_timer += dt
                    if person.ocular_break_timer >= person.gaze_away_limit:
                        person.screen_gaze_accumulation_timer = 0.0
                        person.ocular_break_timer = 0.0
                        person.state = "Tracking Active"
                        person.ocular_break_announced = False

            if person.state in ["Tracking Active", "Posture Deficit Alert", "Ocular Break Recommended"]:
                person.sitting_duration_clock += dt
                if person.sitting_duration_clock >= person.session_limit:
                    person.state = "Session Limit Reached - Stand Up!"
                    if not person.session_limit_announced:
                        print(f"[TIMER ALERT] Triggering Voice Alert for: {person.state}")
                        self._dispatch_voice(f"{person.name}, you have been sitting for too long. Please stand up.")
                        person.session_limit_announced = True

            if person.state == "Posture Deficit Alert":
                person.slouch_timer += dt
                if person.slouch_timer >= 8.0:
                    if not person.slouch_announced:
                        print(f"[TIMER ALERT] Triggering Voice Alert for: {person.state}")
                        self._dispatch_voice(f"Please correct your posture, {person.name}.")
                        person.slouch_announced = True
            else:
                person.slouch_timer = max(0.0, person.slouch_timer - dt)
                person.slouch_announced = False

            if person.state == "Standing":
                person.standing_duration_clock += dt
                person.slouch_timer = max(0.0, person.slouch_timer - dt)
                if person.standing_duration_clock >= person.stand_requirement:
                    person.sitting_duration_clock = 0.0
                    person.session_limit_announced = False
                if is_standing:
                    person.baseline_shoulder_y = (person.baseline_shoulder_y * 0.95) + (shoulder_center_y * 0.05)

            # Forced falling-edge latch at the very end
            is_effectively_standing = is_standing or (is_pinned_to_ceiling and nose_missing)
            if not is_effectively_standing and person.state == "Standing":
                person.state_history_window.clear()
                person.state = "Calibrating"
                person.calibration_start = current_time
                person.calibration_accumulator = []
                person.standing_duration_clock = 0.0
                person.calibration_announced = True
                self._dispatch_voice("Re-calibrating posture workspace.")

            # Adaptive Momentum Anchor
            if person.state == "Tracking Active":
                fast_frames = getattr(person, 'fast_relatch_frames', 0)
                alpha = 0.20 if fast_frames > 0 else 0.005
                person.baseline_shoulder_y = (person.baseline_shoulder_y * (1.0 - alpha)) + (shoulder_center_y * alpha)
                person.baseline_torso_ratio = (person.baseline_torso_ratio * (1.0 - alpha)) + (current_ratio * alpha)
                if fast_frames > 0:
                    person.fast_relatch_frames = fast_frames - 1

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
                matched_person = None
                best_cost = float('inf')
                
                for tid, person in self.tracked_persons.items():
                    tc = person.get_centroid()
                    
                    pixel_distance = np.linalg.norm(np.array(det_centroid) - np.array(tc))
                    if pixel_distance > (w * 0.25):
                        continue
                        
                    scale_aware_dist = pixel_distance / (shoulder_width * w + 1e-6)
                    cost = (scale_aware_dist * 0.5) + (np.linalg.norm(embedding - person.embedding) * 0.5)
                    
                    is_pinned_to_ceiling = (box[1] <= h * 0.05)
                    
                    max_cost_limit = 0.70 if person.state == "Searching / Re-acquiring" else 0.55
                    if is_pinned_to_ceiling:
                        max_cost_limit = max(max_cost_limit, 0.95)
                        
                    if person.state == "Standing" or person.state == "Searching / Re-acquiring":
                        max_cost_limit = max(max_cost_limit, 0.90)
                    elif len(set(person.state_history_window)) > 1:
                        max_cost_limit = max(max_cost_limit, 0.90)
                    
                    if cost <= max_cost_limit and cost < best_cost:
                        best_cost = cost; matched_person = person
                        
                if matched_person is None:
                    profile_name = self._match_profile(embedding)
                    matched_coasting_person = None
                    if profile_name and profile_name not in assigned_names:
                        for tid, person in self.tracked_persons.items():
                            if person.name == profile_name and person.state == "Searching / Re-acquiring":
                                matched_coasting_person = person; break
                                
                    if matched_coasting_person is not None:
                        matched_person = matched_coasting_person
                        matched_person.state = "Tracking Active"
                        matched_person.last_seen = current_time
                        matched_person.last_update = current_time
                        matched_person.recovery_calibration_start = current_time
                        matched_person.recovery_accumulator = []
                    else:
                        track_hash = f"Person_hash_{int(current_time * 1000)}_{self.track_counter}"
                        self.track_counter += 1
                        matched_person = Person(track_hash, embedding, box)
                        self.tracked_persons[track_hash] = matched_person

                dt = current_time - matched_person.last_update
                if matched_person.state == "Searching / Re-acquiring":
                    matched_person.state = "Tracking Active"
                    matched_person.recovery_calibration_start = current_time
                    matched_person.recovery_accumulator = []

                matched_person.last_seen = current_time
                matched_person.last_update = current_time
                matched_person.box = box
                active_ids.add(matched_person.track_id)
                
                matched_person.pitch = (matched_person.pitch * 0.6) + (det["pitch"] * 0.4)
                matched_person.yaw = det["yaw"]
                matched_person.roll = det["roll"]
                
                shoulder_center_y = (pose["left_shoulder"].y + pose["right_shoulder"].y) / 2.0
                current_ratio = np.abs(shoulder_center_y - pose["nose"].y) / max(shoulder_width, 1e-6)
                matched_person.last_y = shoulder_center_y
                
                if matched_person.smoothed_pitch is None:
                    matched_person.smoothed_pitch = matched_person.pitch
                    matched_person.smoothed_ratio = current_ratio
                    matched_person.smoothed_y = shoulder_center_y
                else:
                    alpha = 0.15
                    matched_person.smoothed_pitch = (1 - alpha) * matched_person.smoothed_pitch + alpha * matched_person.pitch
                    matched_person.smoothed_ratio = (1 - alpha) * matched_person.smoothed_ratio + alpha * current_ratio
                    matched_person.smoothed_y = (1 - alpha) * matched_person.smoothed_y + alpha * shoulder_center_y
                
                profile_name = self._match_profile(embedding)
                
                if profile_name:
                    if profile_name in assigned_names:
                        matched_person.name = "Unknown"
                        matched_person.state = "Secondary Bystander"
                        matched_person.calibration_start = None
                        matched_person.calibration_accumulator = []
                    else:
                        assigned_names.add(profile_name)
                        was_unknown = (matched_person.name == "Unknown")
                        matched_person.name = profile_name
                        if was_unknown:
                            print(f"[BIOMETRICS] Face verified as: {matched_person.name}")
                        
                        p = self.profiles[profile_name]
                        matched_person.slouch_sensitivity = p["slouch_sensitivity"]
                        matched_person.session_limit = p["session_limit"]
                        matched_person.stand_requirement = p["stand_requirement"]
                        matched_person.gaze_away_limit = float(p.get("gaze_away_limit", 20.0))
                        matched_person.biometric_cutoff = p.get("biometric_cutoff", 0.55)
                        
                        if was_unknown:
                            matched_person.state = "Calibrating"
                            matched_person.calibration_start = current_time
                            matched_person.calibration_accumulator = []
                            matched_person.calibration_announced = False
                        elif matched_person.state in ["Unregistered Guest", "Searching / Re-acquiring", "Secondary Bystander"]:
                            matched_person.state = "Tracking Active"
                else:
                    if matched_person.name != "Unknown" and matched_person.name in self.profiles:
                        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-6)
                        db_emb = self.profiles[matched_person.name]["embedding"]
                        db_emb_norm = db_emb / (np.linalg.norm(db_emb) + 1e-6)
                        cosine_similarity = np.dot(emb_norm, db_emb_norm)
                        cosine_dist = 1.0 - cosine_similarity
                        if cosine_dist > matched_person.biometric_cutoff:
                            matched_person.calibration_accumulator = []
                            matched_person.calibration_start = None
                            matched_person.name = "Unknown"
                            matched_person.state = "Unregistered Guest"
                
                if matched_person.name == "Unknown":
                    if not is_primary:
                        matched_person.state = "Secondary Bystander"
                    else:
                        matched_person.state = "Unregistered Guest"
                    continue
                    
                gaze_x, is_looking_away = self.inference_engine.extract_pupil_gaze(frame, det["l_eye"], det["r_eye"])
                matched_person.gaze_x = gaze_x
                matched_person.is_looking_away = is_looking_away
                
                self._evaluate_single_target_health(matched_person, pose, current_ratio, dt, current_time, frame_shape)
                
                if matched_person.state != matched_person.last_state:
                    print(f"[STATE CHANGE] {matched_person.name} transitioned from {matched_person.last_state} -> {matched_person.state}")
                    matched_person.last_state = matched_person.state
                
                if current_time - matched_person.last_log_time >= 1.0:
                    matched_person.last_log_time = current_time
                                    
            for tid, person in list(self.tracked_persons.items()):
                if tid not in active_ids:
                    if current_time - person.last_seen > 10.0:
                        if person.name != "Unknown":
                            self.db_manager.log_session_metrics(person.name, "session_ended", person.sitting_duration_clock)
                        del self.tracked_persons[tid]
                    else:
                        if person.state not in ["Calibrating", "Secondary Bystander"]:
                            person.state = "Searching / Re-acquiring"
                    if person.state != person.last_state:
                        print(f"[STATE CHANGE] {person.name} transitioned from {person.last_state} -> {person.state}")
                        person.last_state = person.state