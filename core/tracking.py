import time
import logging
import threading
import numpy as np
import cv2
import math
import uuid
import queue
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import deque, Counter
from core.biometrics import CrossPlatformInferenceManager

logger = logging.getLogger("DeskBotV3.Tracking")
state_mutex = threading.Lock()

try:
    import pyttsx3
    HAVE_PYTTSX3 = True
except ImportError:
    HAVE_PYTTSX3 = False
    print("[!] pyttsx3 not installed. Voice alerts will be muted.")

class VoiceAlertDaemon:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.running = True
        self.last_played = {}
        threading.Thread(target=self._worker, daemon=True).start()
        
    def _worker(self):
        if not HAVE_PYTTSX3:
            return
        import pyttsx3
        try:
            import pythoncom
        except ImportError:
            pythoncom = None
            
        while self.running:
            try:
                msg, category = self.message_queue.get(timeout=1.0)
                try:
                    if pythoncom:
                        pythoncom.CoInitialize()
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 145)
                    engine.say(msg)
                    engine.runAndWait()
                    # Delete engine explicitly to force SAPI release
                    del engine
                except Exception as e:
                    print(f"[VOICE DAEMON ERROR] {e}")
                finally:
                    if pythoncom:
                        try:
                            pythoncom.CoUninitialize()
                        except Exception:
                            pass
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[VOICE DAEMON QUEUE ERROR] {e}")

    def dispatch(self, text: str, category: str, cooldown: float = 30.0) -> None:
        now = time.time()
        last = self.last_played.get(category, 0.0)
        if now - last > cooldown:
            self.last_played[category] = now
            self.message_queue.put((text, category))

class Person:
    """
    Represents a tracked human target containing bounding box logic, biometric identity,
    and temporal state logic for posture evaluation.
    """
    def __init__(self, track_id: int, embedding: np.ndarray, box: list) -> None:
        self.track_id = track_id
        self.embedding = embedding
        self.box = box
        self.name = "Unknown"
        self.state = "Unregistered Guest"
        self.last_state = "Unregistered Guest"
        self.is_verified = False
        self.is_posture_calibrated = False
        
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
        
        self.sustained_slouch_debounce_timer = 0.0
        self.tracking_active_debounce_timer = 0.0
        
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
        self.calibrated_baseline_neck_pitch = 0.0           
        self.baseline_shoulder_y = 0.0      
        self.calibration_accumulator = []  
        self.calibration_start = None
        self.biometric_consensus_frame_counter = 0
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
    """
    Primary tracking and biometric anchor engine. Orchestrates spatial filtering (IoU),
    facial recognition persistence, and single-target posture evaluations.
    """
    def __init__(self, db_manager: Any) -> None:
        """Initializes the health tracking engine and tracking dictionaries."""
        self.mutex = threading.Lock()
        self.db_manager = db_manager
        self.profiles = self.db_manager.load_all_profiles()
        
        self.inference_manager = CrossPlatformInferenceManager()
        self.voice_daemon = VoiceAlertDaemon()
        
        self.historical_users = {}
        for profile_name, profile_data in self.profiles.items():
            dummy_box = np.array([0, 0, 0, 0])
            cold_person = Person(f"Person_hash_cold_{profile_name}", profile_data["embedding"], dummy_box)
            cold_person.name = profile_name
            cold_person.is_posture_calibrated = True
            self.historical_users[profile_name] = cold_person

        self.tracked_persons = {}
        self.track_id_counter = 0
        self.frame_count = 0
        
        self.primary_user_track_id = None
        self.current_authenticated_user = None
        self.anchor_lost_frame_counter = 0
        self.last_voice_alert = 0.0
        self.system_was_manually_cleared = False
        self.manual_recalibration_requested = False

    def sync_profiles(self) -> None:
        """Synchronizes internal trackers with database thresholds."""
        with self.mutex:
            self.profiles = self.db_manager.load_all_profiles()
            for person in self.tracked_persons.values():
                if person.name in self.profiles:
                    profile_config_map = self.profiles[person.name]
                    person.slouch_sensitivity = profile_config_map["slouch_sensitivity"]
                    person.session_limit = profile_config_map["session_limit"]
                    person.stand_requirement = profile_config_map["stand_requirement"]
                    person.gaze_away_limit = float(profile_config_map.get("ocular_break_duration", 20.0))
                    person.screen_gaze_limit = float(profile_config_map.get("screen_gaze_limit", 1200.0))
                    person.biometric_cutoff = profile_config_map.get("biometric_cutoff", 0.55)

    def _match_profile(self, embedding: np.ndarray, spatial_penalty: float = 0.0) -> Tuple[Optional[str], float]:
        """Compares target embedding against registered database profiles."""
        matched_db_profile_string = None
        calculated_similarity = -1.0
        norm_embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        
        for name, profile in self.profiles.items():
            db_emb = profile["embedding"]
            norm_template = db_emb / (np.linalg.norm(db_emb) + 1e-6)
            cosine_similarity = np.dot(norm_embedding, norm_template)
            
            penalized_similarity = cosine_similarity - (spatial_penalty * 0.3)
            if penalized_similarity > calculated_similarity:
                calculated_similarity = penalized_similarity
                matched_db_profile_string = name
                
        validated_profile_name = None
        if calculated_similarity >= 0.86:
            validated_profile_name = matched_db_profile_string
        else:
            validated_profile_name = None
            
        return validated_profile_name, calculated_similarity

    def _dispatch_voice(self, text: str, category: str = "general", cooldown: float = 10.0) -> None:
        """Dispatches an asynchronous voice alert through the dedicated daemon queue."""
        if hasattr(self, 'voice_daemon'):
            self.voice_daemon.dispatch(text, category, cooldown)

    def _evaluate_single_target_health(self, person: "Person", pose: dict, current_ratio: float, dt: float, current_time: float, frame_shape: tuple) -> None:
        """
        Evaluates frame-level posture thresholds against calibrated baselines.
        Applies strict temporal hysteresis to prevent state chatter.
        """
        if (person.name.startswith("Unknown") or self.primary_user_track_id is None) and person.state != "Calibrating":
            person.state = "Unregistered Target"
            person.sustained_slouch_debounce_timer = 0.0
            person.tracking_active_debounce_timer = 0.0
            return

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
                
            person.calibration_accumulator.append((person.pitch, current_ratio_computed, shoulder_center_y))
            
            if elapsed_calib >= 5.0:
                if person.calibration_accumulator:
                    person.calibrated_baseline_neck_pitch = float(np.mean([i[0] for i in person.calibration_accumulator]))
                    person.baseline_torso_ratio = float(np.mean([i[1] for i in person.calibration_accumulator]))
                    person.baseline_shoulder_y = float(np.mean([i[2] for i in person.calibration_accumulator]))
                person.calibration_accumulator = []
                person.state = "Tracking Active"
                if self.profiles:
                    person.name = list(self.profiles.keys())[0]
                else:
                    person.name = f"Operator #{str(person.track_id)[-4:]}"
                person.is_posture_calibrated = True
                self.primary_user_track_id = person.track_id
                self._dispatch_voice("Calibration successful. Posture monitoring is now active.", "calibration_success", 30.0)
            return

        if person.state in ["Tracking Active", "Standing", "Looking Away", "Ocular Break Recommended", "Session Limit Reached - Stand Up!", "Posture Deficit Alert"]:
            
            if not person.is_looking_away:
                person.screen_gaze_accumulation_timer += dt
                if person.state != "Ocular Break Recommended":
                    person.ocular_break_timer = 0.0
            
            shoulder_width = np.abs(pose['right_shoulder'].x - pose['left_shoulder'].x)
            shoulder_center_y = (pose['left_shoulder'].y + pose['right_shoulder'].y) / 2.0
            
            current_neck_pitch_angle = person.smoothed_pitch if person.smoothed_pitch is not None else person.pitch
            current_torso_depth_ratio = person.smoothed_ratio if person.smoothed_ratio is not None else current_ratio

            relative_slouch = current_neck_pitch_angle - person.calibrated_baseline_neck_pitch
            
            if person.state == "Posture Deficit Alert":
                is_fully_upright = (current_torso_depth_ratio >= person.baseline_torso_ratio * 0.95) and (relative_slouch <= person.slouch_sensitivity * 0.40)
                is_slouching = not is_fully_upright
            else:
                is_slouching = (current_torso_depth_ratio < (person.baseline_torso_ratio * 0.80)) or (relative_slouch > 22.0)
            
            normalized_height_delta = (person.baseline_shoulder_y - shoulder_center_y) / max(shoulder_width, 1e-6)
            is_standing = (normalized_height_delta > 0.65) or (current_torso_depth_ratio > (person.baseline_torso_ratio * 1.65))
            
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

            if len(person.state_history_window) >= 25:
                counter = Counter(person.state_history_window)
                if person.state == "Posture Deficit Alert":
                    if counter.get("Tracking Active", 0) >= 20:
                        filtered_candidate = "Tracking Active"
                    else:
                        filtered_candidate = "Posture Deficit Alert"
                else:
                    filtered_candidate = counter.most_common(1)[0][0]
            else:
                filtered_candidate = frame_candidate
                
            if filtered_candidate == "Posture Deficit Alert":
                person.sustained_slouch_debounce_timer += dt
                person.tracking_active_debounce_timer = 0.0
                if person.sustained_slouch_debounce_timer >= 3.0:
                    person.state = "Posture Deficit Alert"
            elif filtered_candidate == "Tracking Active":
                person.tracking_active_debounce_timer += dt
                person.sustained_slouch_debounce_timer = 0.0
                if person.tracking_active_debounce_timer >= 1.5:
                    person.state = "Tracking Active"
            elif filtered_candidate == "Standing":
                person.state = "Standing"
                person.sustained_slouch_debounce_timer = 0.0
                person.tracking_active_debounce_timer = 0.0

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
                if not person.slouch_announced:
                    print(f"[TIMER ALERT] Triggering Voice Alert for: {person.state}")
                    self._dispatch_voice(f"Please correct your posture, {person.name}.", category="posture_alert", cooldown=30.0)
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

            is_effectively_standing = is_standing or (is_pinned_to_ceiling and nose_missing)
            if not is_effectively_standing and person.state == "Standing":
                person.state_history_window.clear()
                person.state = "Calibrating"
                person.calibration_start = current_time
                person.calibration_accumulator = []
                person.standing_duration_clock = 0.0
                person.calibration_announced = True
                self._dispatch_voice("Re-calibrating posture workspace.")

            if person.state == "Tracking Active":
                fast_frames = getattr(person, 'fast_relatch_frames', 0)
                alpha = 0.20 if fast_frames > 0 else 0.005
                person.baseline_shoulder_y = (person.baseline_shoulder_y * (1.0 - alpha)) + (shoulder_center_y * alpha)
                person.baseline_torso_ratio = (person.baseline_torso_ratio * (1.0 - alpha)) + (current_ratio * alpha)
                if fast_frames > 0:
                    person.fast_relatch_frames = fast_frames - 1

    def _calculate_iou(self, boxA: list, boxB: list) -> float:
        """Calculates Intersection over Union for bounding box suppression."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def process_frame_mot(self, frame: np.ndarray, frame_shape: tuple) -> Any:
        """
        Executes multi-object tracking and health evaluations on each incoming frame.
        """
        detections = self.inference_manager.execute_stage1_detector(frame)
        self.frame_count += 1
        is_database_empty = (len(self.profiles) == 0)
        current_time = time.time()
        h, w = frame_shape[:2]
        active_ids = set()
        frame_level_locked_identities = set()
        
        ranked_detections = []
        for det in detections:
            box = det["box"]
            centroid_x = (box[0] + box[2]) / 2.0
            shoulder_width = np.abs(det["pose"]["left_shoulder"].x - det["pose"]["right_shoulder"].x)
            normalized_cosine_similarity = shoulder_width - (np.abs(centroid_x - w/2) / w)
            ranked_detections.append((normalized_cosine_similarity, det))
        ranked_detections.sort(key=lambda x: x[0], reverse=True)
        
        filtered_ranked_detections = []
        for det_tup in ranked_detections:
            normalized_cosine_similarity, det = det_tup
            box = det["box"]
            is_duplicate = False
            for f_score, f_det in filtered_ranked_detections:
                if self._calculate_iou(box, f_det["box"]) > 0.40:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_ranked_detections.append(det_tup)
                
        ranked_detections = filtered_ranked_detections
        
        with self.mutex:
            primary_det = ranked_detections[0][1] if ranked_detections else None
            
            for normalized_cosine_similarity, det in ranked_detections:
                det["matched_person"] = None
                
            for normalized_cosine_similarity, det in ranked_detections:
                box = det["box"]
                pose = det["pose"]
                det_centroid = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                shoulder_width = np.abs(pose["left_shoulder"].x - pose["right_shoulder"].x)
                
                matched_person = None
                best_cost = float('inf')
                
                for track_id, person in self.tracked_persons.items():
                    if person.name == "Unknown" or track_id in active_ids:
                        continue
                        
                    tc = person.get_centroid()
                    pixel_distance = np.linalg.norm(np.array(det_centroid) - np.array(tc))
                    if pixel_distance > (w * 0.25):
                        continue
                        
                    scale_aware_dist = pixel_distance / (shoulder_width * w + 1e-6)
                    cost = scale_aware_dist
                    
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
                        
                if matched_person is not None:
                    det["matched_person"] = matched_person
                    frame_level_locked_identities.add(matched_person.name)
                    active_ids.add(matched_person.track_id)

            for normalized_cosine_similarity, det in ranked_detections:
                matched_person = det["matched_person"]
                
                box = det["box"]
                pose = det["pose"]
                det_centroid = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                shoulder_width = np.abs(pose["left_shoulder"].x - pose["right_shoulder"].x)
                
                if matched_person is None:
                    best_cost = float('inf')
                    for track_id, person in self.tracked_persons.items():
                        if person.name != "Unknown" or track_id in active_ids:
                            continue
                            
                        tc = person.get_centroid()
                        pixel_distance = np.linalg.norm(np.array(det_centroid) - np.array(tc))
                        if pixel_distance > (w * 0.25):
                            continue
                            
                        scale_aware_dist = pixel_distance / (shoulder_width * w + 1e-6)
                        cost = scale_aware_dist
                        
                        max_cost_limit = 0.70 if person.state == "Searching / Re-acquiring" else 0.55
                        if cost <= max_cost_limit and cost < best_cost:
                            best_cost = cost; matched_person = person
                            
                    if matched_person is None:
                        track_hash = f"Person_hash_{int(current_time * 1000)}_{self.track_id_counter}"
                        self.track_id_counter += 1
                        dummy_embedding = np.zeros(128, dtype=np.float32)
                        matched_person = Person(track_hash, dummy_embedding, box)
                        matched_person.pose = pose
                        self.tracked_persons[track_hash] = matched_person
                    
                    active_ids.add(matched_person.track_id)
                    
                dt = current_time - matched_person.last_update
                if matched_person.state == "Searching / Re-acquiring":
                    matched_person.state = "Tracking Active"
                    matched_person.recovery_calibration_start = current_time
                    matched_person.recovery_accumulator = []

                matched_person.last_seen = current_time
                matched_person.last_update = current_time
                matched_person.box = box
                matched_person.pose = pose
                
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
                    alpha = 0.10
                    matched_person.smoothed_pitch = (1 - alpha) * matched_person.smoothed_pitch + alpha * matched_person.pitch
                    matched_person.smoothed_ratio = (1 - alpha) * matched_person.smoothed_ratio + alpha * current_ratio
                    matched_person.smoothed_y = (1 - alpha) * matched_person.smoothed_y + alpha * shoulder_center_y
                
                if is_database_empty:
                    matched_person.name = "Unknown (Ready for Registration)"
                    if np.all(matched_person.embedding == 0):
                        matched_person.embedding = self.inference_manager.execute_stage2_biometrics(det["roi_frame"])
                    if matched_person.state in ["Unregistered Guest", "Secondary Bystander", "Searching / Re-acquiring"]:
                        if not matched_person.is_posture_calibrated:
                            matched_person.state = "Calibrating"
                            matched_person.calibration_start = current_time
                            matched_person.calibration_accumulator = []
                            matched_person.calibration_announced = False
                else:
                    run_global_biometric_search = False
                    
                    if self.primary_user_track_id is not None:
                        if matched_person.track_id == self.primary_user_track_id:
                            # Clear searching state immediately upon re-acquisition
                            if matched_person.state in ["Searching / Re-acquiring", "Absent"]:
                                embedding = self.inference_manager.execute_stage2_biometrics(det["roi_frame"])
                                validated_profile_name, calculated_similarity = self._match_profile(embedding)
                                
                                if validated_profile_name is not None and validated_profile_name == self.current_authenticated_user:
                                    print(f"[BIOMETRICS] Face matched via LBP signature (sim: {calculated_similarity:.2f}). Resuming track.")
                                    matched_person.state = "Tracking Active"
                                    matched_person.state_history_window.clear()
                                else:
                                    continue
                                
                            if self.manual_recalibration_requested:
                                matched_person.is_posture_calibrated = False
                                matched_person.state = "Calibrating"
                                matched_person.calibration_start = current_time
                                matched_person.calibration_accumulator = []
                                matched_person.calibration_announced = False
                                self.manual_recalibration_requested = False
                        else:
                            primary_person = self.tracked_persons.get(self.primary_user_track_id)
                            if primary_person is None:
                                self.primary_user_track_id = None
                                run_global_biometric_search = True
                            elif primary_person.state in ["Searching / Re-acquiring", "Absent"]:
                                crop_y1, crop_y2 = int(max(0, box[1])), int(min(h, box[3]))
                                crop_x1, crop_x2 = int(max(0, box[0])), int(min(w, box[2]))
                                if (crop_y2 - crop_y1) < 64 or (crop_x2 - crop_x1) < 64:
                                    matched_person.name = "Unknown / Bystander"
                                    matched_person.state = "Secondary Bystander"
                                    continue
                                    
                                embedding = self.inference_manager.execute_stage2_biometrics(det["roi_frame"])
                                norm_emb = embedding / (np.linalg.norm(embedding) + 1e-6)
                                norm_primary = primary_person.embedding / (np.linalg.norm(primary_person.embedding) + 1e-6)
                                sim = np.dot(norm_emb, norm_primary)
                                
                                if sim >= 0.86:
                                    print(f"[BIOMETRICS] Returning target matched (sim: {sim:.2f}). Re-latching anchor.")
                                    old_person = primary_person
                                    old_person.track_id = matched_person.track_id
                                    old_person.box = matched_person.box
                                    old_person.pose = matched_person.pose
                                    old_person.last_seen = current_time
                                    old_person.last_update = current_time
                                    
                                    if self.primary_user_track_id in self.tracked_persons:
                                        del self.tracked_persons[self.primary_user_track_id]
                                        
                                    self.tracked_persons[matched_person.track_id] = old_person
                                    matched_person = old_person
                                    self.primary_user_track_id = matched_person.track_id
                                    matched_person.state = "Tracking Active"
                                    matched_person.state_history_window.clear()
                                else:
                                    matched_person.name = "Unknown / Bystander"
                                    matched_person.state = "Secondary Bystander"
                                    continue
                            else:
                                matched_person.name = "Unknown / Bystander"
                                matched_person.state = "Secondary Bystander"
                                continue
                    else:
                        run_global_biometric_search = True

                    if run_global_biometric_search:
                        crop_y1, crop_y2 = int(max(0, box[1])), int(min(h, box[3]))
                        crop_x1, crop_x2 = int(max(0, box[0])), int(min(w, box[2]))
                        crop_h = crop_y2 - crop_y1
                        crop_w = crop_x2 - crop_x1
                        
                        if crop_h < 64 or crop_w < 64:
                            matched_person.name = "Unknown / Bystander"
                            matched_person.state = "Secondary Bystander"
                            matched_person.biometric_consensus_frame_counter = 0
                            continue
                            
                        if self.primary_user_track_id is None or self.manual_recalibration_requested:
                            embedding = self.inference_manager.execute_stage2_biometrics(det["roi_frame"])
                            matched_person.embedding = embedding
                            
                            dist_from_center = np.linalg.norm(np.array(det_centroid) - np.array([w/2, h/2]))
                            spatial_penalty = dist_from_center / (w + 1e-6)
                            
                            validated_profile_name, calculated_similarity = self._match_profile(embedding, spatial_penalty)
                            
                            if validated_profile_name is not None and validated_profile_name in self.historical_users:
                                old_person = self.historical_users[validated_profile_name]
                                old_person.track_id = matched_person.track_id
                                old_person.box = matched_person.box
                                old_person.pose = matched_person.pose
                                old_person.last_seen = current_time
                                old_person.last_update = current_time
                                self.tracked_persons[matched_person.track_id] = old_person
                                matched_person = old_person
                                
                                self.primary_user_track_id = matched_person.track_id
                                self.current_authenticated_user = validated_profile_name
                                matched_person.name = validated_profile_name
                                
                                # Force immediate promotion out of default guest status
                                matched_person.state = "Tracking Active"
                                matched_person.is_posture_calibrated = True
                            elif validated_profile_name is None:
                                matched_person.biometric_consensus_frame_counter = 0
                                matched_person.name = "Unknown / Bystander"
                                matched_person.state = "Calibrating"
                                matched_person.is_posture_calibrated = False
                                matched_person.calibration_start = current_time
                                matched_person.calibration_accumulator = []
                                continue
                            else:
                                matched_person.biometric_consensus_frame_counter += 1
                                if matched_person.biometric_consensus_frame_counter >= 15:
                                    self.primary_user_track_id = matched_person.track_id
                                    matched_person.name = validated_profile_name
                                    profile_config_map = self.profiles[validated_profile_name]
                                    matched_person.slouch_sensitivity = profile_config_map["slouch_sensitivity"]
                                    matched_person.session_limit = profile_config_map["session_limit"]
                                    matched_person.stand_requirement = profile_config_map["stand_requirement"]
                                    matched_person.gaze_away_limit = float(profile_config_map.get("gaze_away_limit", 20.0))
                                    matched_person.biometric_cutoff = profile_config_map.get("biometric_cutoff", 0.55)
                                    
                                    if not matched_person.is_posture_calibrated:
                                        matched_person.state = "Calibrating"
                                        matched_person.calibration_start = current_time
                                        matched_person.calibration_accumulator = []
                                        matched_person.calibration_announced = False
                                    else:
                                        matched_person.state = "Tracking Active"
                                else:
                                    matched_person.name = "Unknown"
                                    matched_person.state = "Unregistered Guest"
                                    continue
                        else:
                            if matched_person.name != self.primary_user_track_id and matched_person.name == "Unknown":
                                matched_person.name = "Unknown / Bystander"
                                matched_person.state = "Secondary Bystander"
                                continue

                # The only track that reaches here is the primary user anchor track.
                gaze_x, is_looking_away = self.inference_manager.extract_pupil_gaze(frame, det["l_eye"], det["r_eye"])
                matched_person.gaze_x = gaze_x
                matched_person.is_looking_away = is_looking_away
                
                self._evaluate_single_target_health(matched_person, pose, current_ratio, dt, current_time, frame_shape)
                
                if matched_person.state != getattr(matched_person, "last_state", "Unknown"):
                    print(f"[STATE CHANGE] {matched_person.name} transitioned from {getattr(matched_person, 'last_state', 'Unknown')} -> {matched_person.state}")
                    matched_person.last_state = matched_person.state
                
                if current_time - matched_person.last_log_time >= 1.0:
                    matched_person.last_log_time = current_time
                                    
            # =================================================================
            # ✅ CORRECTED: HARD SESSION EXPIRATION & RESET WITH ESCAPE BRAKE
            # =================================================================
            if not hasattr(self, "anchor_lost_frame_counter"):
                self.anchor_lost_frame_counter = 0
                
            if self.primary_user_track_id is not None:
                if self.primary_user_track_id not in active_ids:
                    self.anchor_lost_frame_counter += 1
                    eviction_limit = 150
                            
                    if self.anchor_lost_frame_counter >= eviction_limit:
                        if self.primary_user_track_id in self.tracked_persons:
                            primary_person = self.tracked_persons[self.primary_user_track_id]
                            
                            # Only trigger logs and state transitions if we aren't already searching!
                            if primary_person.state != "Searching / Re-acquiring":
                                print(f"[BIOMETRICS] Primary anchor lost for {eviction_limit} frames. Soft-timeout triggered.")
                                primary_person.state = "Searching / Re-acquiring"
                                primary_person.state_history_window.clear()
                                # Stash the last known authenticated workspace token name
                                self.current_authenticated_user = primary_person.name
                        
                        # Break out and reset the counter so it stops printing and allows camera thread recovery
                        self.anchor_lost_frame_counter = 0
                else:
                    self.anchor_lost_frame_counter = 0
                                    
            for track_id, person in list(self.tracked_persons.items()):
                if track_id not in active_ids:
                    if current_time - person.last_seen > 10.0:
                        if person.state != "Absent":
                            if person.name != "Unknown" and person.name != "Unknown / Bystander":
                                self.db_manager.log_session_metrics(person.name, "session_ended", person.sitting_duration_clock)
                            person.state = "Absent"
                            person.state_history_window.clear()
                    else:
                        if person.state not in ["Calibrating", "Secondary Bystander", "Absent"]:
                            person.state = "Searching / Re-acquiring"
                    if person.state != getattr(person, "last_state", "Unknown"):
                        print(f"[STATE CHANGE] {person.name} transitioned from {getattr(person, 'last_state', 'Unknown')} -> {person.state}")
                        person.last_state = person.state

