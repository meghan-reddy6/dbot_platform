from collections import Counter
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

class VoiceAlertDaemon:
    def __init__(self):
        self.last_alert_time = 0.0
        self.alert_cooldown = 20.0
        self.audio_thread_active = False

    def _say_via_subprocess(self, text_prompt: str):
        def worker():
            try:
                import subprocess
                import sys
                import os
                if sys.platform == "win32":
                    ps_script = f"Add-Type -AssemblyName System.speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('{text_prompt.replace(chr(39), chr(39)+chr(39))}')"
                    subprocess.Popen(["powershell", "-Command", ps_script], creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    os.system(f"espeak '{text_prompt}' &")
            except Exception as e:
                print(f"[VOICE DAEMON ERROR] {e}")
            finally:
                self.audio_thread_active = False
                
        threading.Thread(target=worker, daemon=True).start()

    def dispatch(self, text: str, category: str = "general", cooldown: float = 30.0) -> None:
        import time
        current_time = time.time()
        is_cooldown_passed = (current_time - self.last_alert_time > self.alert_cooldown) or (category == "registration")
        if is_cooldown_passed and not self.audio_thread_active:
            self.last_alert_time = current_time
            self.audio_thread_active = True
            self._say_via_subprocess(text)

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
        self.verification_status = "UNKNOWN"
        self.biometric_match_counter = 0
        self.candidate_name = None
        self.verified_name = None
        self.frame_val_name = None
        self.verification_timer = 0.0
        self.lost_grace_timer = 0.0
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
        self.last_analytics_flush_time = 0.0
        
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
        import os
        import json
        self.profiles_json_path = os.path.abspath('profiles_cache.json')
        self.profiles = self.db_manager.load_all_profiles()
        
        if os.path.exists(self.profiles_json_path):
            with open(self.profiles_json_path, 'r') as f:
                saved_cache = json.load(f)
                for name, vectors in saved_cache.items():
                    if name not in self.profiles:
                        self.profiles[name] = {}
                    self.profiles[name]["embeddings"] = vectors
                print(f"[SYSTEM BOOT] Cold-boot telemetry successful! Loaded {len(saved_cache)} profiles from {self.profiles_json_path}")
        else:
            print(f"[SYSTEM BOOT] No existing profile cache found at {self.profiles_json_path}")
        
        self.inference_manager = CrossPlatformInferenceManager()
        self.voice_daemon = VoiceAlertDaemon()
        
        self.historical_users = {}
        for profile_name, profile_data in self.profiles.items():
            dummy_box = np.array([0, 0, 0, 0])
            first_emb = profile_data.get("embeddings", [None])[0]
            if first_emb is None and "embedding" in profile_data:
                first_emb = profile_data["embedding"]
            cold_person = Person(f"Person_hash_cold_{profile_name}", first_emb, dummy_box)
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
        self.pending_registration_name = None
        self.trigger_recalibration = False
        self._enrollment_flight_lock = False

    def _load_profiles_json(self):
        import json
        import os
        try:
            if not os.path.exists(self.profiles_json_path):
                raise FileNotFoundError(f"{self.profiles_json_path} does not exist")
                
            with open(self.profiles_json_path, 'r') as f:
                data = json.load(f)
                
            if not data:
                raise ValueError("JSON file is empty")
                
            for name, vectors in data.items():
                if name in self.profiles:
                    self.profiles[name]["embeddings"] = [np.array(v, dtype=np.float32) for v in vectors]
        except Exception as e:
            self.profiles = {}
            print(f"[PERSISTENCE] No valid profile database found. Starting with a clean slate. ({e})")

    def _save_profiles_json(self):
        import json
        import threading
        import os
        
        def bg_save():
            try:
                data = {}
                with self.mutex:
                    for name, profile in self.profiles.items():
                        if "embeddings" in profile:
                            data[name] = [(v.tolist() if isinstance(v, np.ndarray) else v) for v in profile["embeddings"]]
                        elif "embedding" in profile:
                            emb = profile["embedding"]
                            data[name] = [emb.tolist() if isinstance(emb, np.ndarray) else emb]
                
                target_path = getattr(self, 'cache_path', getattr(self, 'profiles_json_path', 'profiles_cache.json'))
                absolute_path = os.path.abspath(target_path)
                
                os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
                with open(absolute_path, 'w') as f:
                    json.dump(data, f)
                    
                print(f"[PERSISTENCE] Hard disk write SUCCESSFUL! File generated at: {absolute_path}")
            except Exception as e:
                print(f"[PERSISTENCE] Background save failed: {e}")
                
        threading.Thread(target=bg_save, daemon=True).start()

    def sync_profiles(self) -> None:
        """Synchronizes internal trackers with database thresholds."""
        with self.mutex:
            self.profiles = self.db_manager.load_all_profiles()
            self._load_profiles_json()
            for person in self.tracked_persons.values():
                if person.name in self.profiles:
                    profile_config_map = self.profiles[person.name]
                    person.slouch_sensitivity = profile_config_map["slouch_sensitivity"]
                    person.session_limit = profile_config_map["session_limit"]
                    person.stand_requirement = profile_config_map["stand_requirement"]
                    person.gaze_away_limit = float(profile_config_map.get("ocular_break_duration", 20.0))
                    person.screen_gaze_limit = float(profile_config_map.get("screen_gaze_limit", 1200.0))
                    person.biometric_cutoff = profile_config_map.get("biometric_cutoff", 0.55)

    def _match_profile(self, embedding: np.ndarray, spatial_penalty: float = 0.0, box: list = None, w: int = 1920) -> Tuple[Optional[str], float]:
        """Compares target embedding against registered database profiles using temporal consensus logic."""
        matched_db_profile_string = None
        highest_consensus = -1.0
        norm_embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
        
        for name, profile in self.profiles.items():
            db_embs = profile.get("embeddings", [])
            if "embedding" in profile and len(db_embs) == 0:
                db_embs = [profile["embedding"]]
                
            if len(db_embs) == 0:
                continue
                
            sim_scores = []
            for db_emb in db_embs:
                db_emb_np = np.array(db_emb, dtype=np.float32)
                norm_template = db_emb_np / (np.linalg.norm(db_emb_np) + 1e-6)
                sim_scores.append(np.dot(norm_embedding, norm_template))
                
            top_score = max(sim_scores)
            mean_score = sum(sim_scores) / len(sim_scores)
            consensus_score = (top_score * 0.7) + (mean_score * 0.3)
            
            if consensus_score > highest_consensus:
                highest_consensus = consensus_score
                matched_db_profile_string = name
                    
        if highest_consensus >= 0.75:
            validated_profile_name = matched_db_profile_string
        else:
            validated_profile_name = "Unknown"
            
        return validated_profile_name, highest_consensus

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
                    person.posture_baseline = float(np.mean([i[0] for i in person.recovery_accumulator]))
                    person.baseline_shoulder_y = float(np.mean([i[1] for i in person.recovery_accumulator]))
                person.recovery_calibration_start = None
                person.recovery_accumulator = []
            return

        if person.state == "Calibrating":
            if getattr(person, 'calibration_start_time', None) is None:
                person.calibration_start_time = current_time
                person.calibration_accumulator = []
                person.calibration_pitch_acc = []
                person.calibration_y_acc = []
                person.is_posture_calibrated = False
            
            if not getattr(person, 'calibration_announced', False):
                person.calibration_announced = True
                self._dispatch_voice(f"{person.name}, please look straight ahead to calibrate your posture baseline.")
                
            elapsed_calib = current_time - person.calibration_start_time
            shoulder_width = np.abs(pose['right_shoulder'].x - pose['left_shoulder'].x)
            shoulder_center_y = (pose['left_shoulder'].y + pose['right_shoulder'].y) / 2.0
            current_ratio_computed = np.abs(shoulder_center_y - pose['nose'].y) / max(shoulder_width, 1e-6)
            
            if current_time - getattr(person, 'last_log_time', 0.0) >= 1.0:
                person.last_log_time = current_time
                
            if not hasattr(person, 'calibration_accumulator'): person.calibration_accumulator = []
            if not hasattr(person, 'calibration_pitch_acc'): person.calibration_pitch_acc = []
            if not hasattr(person, 'calibration_y_acc'): person.calibration_y_acc = []
                
            person.calibration_accumulator.append(current_ratio_computed)
            person.calibration_pitch_acc.append(person.pitch)
            person.calibration_y_acc.append(shoulder_center_y)
            
            if elapsed_calib >= 3.0:
                if len(person.calibration_accumulator) >= 5:
                    person.posture_baseline = float(np.mean(person.calibration_accumulator))
                    person.calibrated_baseline_neck_pitch = float(np.mean(person.calibration_pitch_acc))
                    person.baseline_shoulder_y = float(np.mean(person.calibration_y_acc))
                    person.is_posture_calibrated = True
                else:
                    if not hasattr(person, 'posture_baseline'):
                        person.calibrated_baseline_neck_pitch = 0.0
                        person.posture_baseline = 0.50
                        person.baseline_shoulder_y = float(np.mean(person.calibration_y_acc)) if hasattr(person, 'calibration_y_acc') and person.calibration_y_acc else 0.50
                    person.is_posture_calibrated = True
                    print(f"[CALIBRATION] Timeout fallback triggered. Preserving/Forcing track active for {person.name}.")
                    
                person.calibration_accumulator = []
                person.calibration_pitch_acc = []
                person.calibration_y_acc = []
                person.state = "Tracking Active"
                        
                if person.name in self.profiles:
                    if "embeddings" not in self.profiles[person.name]:
                        self.profiles[person.name]["embeddings"] = [self.profiles[person.name]["embedding"]]
                    self.profiles[person.name]["embeddings"].append(person.embedding)
                    self._save_profiles_json()
                    print(f"[BIOMETRICS] Appended new accessory embedding to profile cluster for {person.name}")
                        
                self.primary_user_track_id = person.track_id
                self._dispatch_voice(f"Calibration successful for {person.name}. Posture monitoring is now active.", "calibration_success", 30.0)
            
            # SECURE THE CALIBRATION HOLD PARAMETERS: Completely block downstream state evaluation
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
            
            if getattr(person, 'calibrated_baseline_neck_pitch', 0.0) == 0.0:
                person.calibrated_baseline_neck_pitch = current_neck_pitch_angle

            relative_slouch = current_neck_pitch_angle - person.calibrated_baseline_neck_pitch
            
            # Make sure posture_baseline is initialized if skipping calibration
            if not hasattr(person, 'posture_baseline'):
                person.posture_baseline = 0.5
            
            if person.state == "Posture Deficit Alert":
                is_fully_upright = (current_torso_depth_ratio >= person.posture_baseline * 0.95) and (relative_slouch <= person.slouch_sensitivity * 0.40)
                is_slouching = not is_fully_upright
            else:
                is_slouching = (current_torso_depth_ratio < (person.posture_baseline * 0.80)) or (relative_slouch > 35.0)
            
            # CORRECT THE GEOMETRIC SITTING VS STANDING BOUNDS
            normalized_height_delta = (person.baseline_shoulder_y - shoulder_center_y) / max(shoulder_width, 1e-6)
            is_standing = (normalized_height_delta > 0.45) or (current_torso_depth_ratio > (person.posture_baseline * 1.85))
            
            is_pinned_to_ceiling = (person.box[1] <= frame_shape[0] * 0.05)
            nose_missing = (pose['nose'].y <= 0.01)
            
            person.is_standing = is_standing

            if not hasattr(person, 'standing_accumulator_time'): person.standing_accumulator_time = 0.0
            if not hasattr(person, 'slouch_accumulator_time'): person.slouch_accumulator_time = 0.0
            if not hasattr(person, 'active_accumulator_time'): person.active_accumulator_time = 0.0

            is_standing_frame = is_standing or (is_pinned_to_ceiling and nose_missing)
            is_slouching_frame = is_slouching and not is_standing_frame

            if is_standing_frame:
                person.standing_accumulator_time += dt
                person.slouch_accumulator_time = 0.0
                person.active_accumulator_time = 0.0
            elif is_slouching_frame:
                person.slouch_accumulator_time += dt
                person.standing_accumulator_time = 0.0
                person.active_accumulator_time = 0.0
            else:
                person.active_accumulator_time += dt
                person.standing_accumulator_time = 0.0
                person.slouch_accumulator_time = 0.0

            if person.standing_accumulator_time >= 2.5:
                person.state = "Standing"
            elif person.slouch_accumulator_time >= 4.0:
                person.state = "Posture Deficit Alert"
            elif person.active_accumulator_time >= 1.5:
                if person.state in ["Standing", "Posture Deficit Alert"]:
                    person.state = "Tracking Active"

            if person.screen_gaze_accumulation_timer >= person.screen_gaze_limit:
                person.state = "Ocular Break Recommended"
                if not person.ocular_break_announced:
                    print(f"[TIMER ALERT] Triggering Voice Alert for: {person.state}")
                    self._dispatch_voice(f"{person.name}, attention, eye strain warning. Please look away from the screen.")
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

            if person.state == "Tracking Active":
                fast_frames = getattr(person, 'fast_relatch_frames', 0)
                alpha = 0.20 if fast_frames > 0 else 0.005
                person.baseline_shoulder_y = (person.baseline_shoulder_y * (1.0 - alpha)) + (shoulder_center_y * alpha)
                person.posture_baseline = (person.posture_baseline * (1.0 - alpha)) + (current_ratio * alpha)
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
        import cv2
        import time
        import math
        import numpy as np
        

        detections = self.inference_manager.execute_stage1_detector(frame)
        self.frame_count += 1
        current_time = time.time()
        h, w = frame_shape[:2]
        
        # --- CONSUME COMMANDS SAFELY INSIDE THE TRACKING CYCLE ---
        with self.mutex:
            if getattr(self, 'pending_registration_name', None) is not None:
                name = self.pending_registration_name
                person_to_register = None
                for tracked_person in self.tracked_persons.values():
                    if tracked_person.name == "Unknown (Ready for Registration)" or "Unknown" in tracked_person.name:
                        person_to_register = tracked_person
                        break
                        
                if person_to_register is not None and hasattr(person_to_register, 'embedding'):
                    try:
                        self.profiles[name] = {
                            "embeddings": [person_to_register.embedding.tolist()],
                            "registration_timestamp": current_time,
                            "slouch_sensitivity": 15.0,
                            "session_limit": 1200,
                            "biometric_cutoff": 0.35,
                            "stand_requirement": 180,
                            "ocular_break_duration": 20,
                            "screen_gaze_limit": 1200
                        }
                        person_to_register.name = name
                        person_to_register.verified_name = name
                        person_to_register.verification_status = "VERIFIED"
                        person_to_register.state = "Tracking Active"
                        person_to_register.lost_grace_timer = 0.0
                        self._save_profiles_json()
                        self._dispatch_voice(f"Registration complete for {name}. Instant 1-frame profile locked.", category="registration")
                        print(f"[TRACKING] Instant 1-frame registration complete for {name}")
                    except Exception as e:
                        print(f"[TRACKING] Failed instant registration: {e}")
                self.pending_registration_name = None
                
            # Unified trigger flag consolidating manual UI callbacks
            if getattr(self, 'trigger_recalibration', False) or getattr(self, 'manual_recalibration_requested', False):
                if self.primary_user_track_id in self.tracked_persons:
                    person = self.tracked_persons[self.primary_user_track_id]
                    person.calibration_start_time = time.time()
                    person.calibration_accumulator = []
                    person.calibration_pitch_acc = []
                    person.calibration_y_acc = []
                    person.state = "Calibrating"
                    person.is_posture_calibrated = False
                    person.calibration_announced = False
                    print(f"[TRACKING] Safely triggered recalibration for {person.name}")
                self.trigger_recalibration = False
                self.manual_recalibration_requested = False
        
        # --- 1. FORCE AN EMPTY FRAME FLUSH GATE ---
        if detections is None or len(detections) == 0:
            with self.mutex:
                keys_to_delete = []
                for track_id, person in self.tracked_persons.items():
                    # Set all previously active primary/authenticated users to Absent
                    if person.name == self.current_authenticated_user or track_id == self.primary_user_track_id:
                        if person.state != "Absent":
                            person.state = "Absent"
                            if hasattr(person, "state_history_window"):
                                person.state_history_window.clear()
                            if person.name and person.name != "Unknown (Ready for Registration)":
                                self.db_manager.log_session_metrics(person.name, "session_ended", person.sitting_duration_clock)
                            if person.state != getattr(person, "last_state", "Unknown"):
                                print(f"[STATE CHANGE] {person.name} transitioned from {getattr(person, 'last_state', 'Unknown')} -> {person.state}")
                                person.last_state = person.state
                    else:
                        keys_to_delete.append(track_id)
                
                for k in keys_to_delete:
                    del self.tracked_persons[k]
            return

        # --- 3. BIOMETRIC GATEKEEPER PASS & DEDICATED ALIGNMENT ---
        if not hasattr(self, 'face_net'):
            import os
            import onnxruntime as ort
            model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
            self.face_net = ort.InferenceSession(os.path.join(model_dir, 'face_detector.onnx'), providers=['CPUExecutionProvider'])
            self.landmark_net = ort.InferenceSession(os.path.join(model_dir, 'face_landmark_detector.onnx'), providers=['CPUExecutionProvider'])

        active_ids = set()
        unmatched_detections = []
        
        # Phase A: Extract embeddings for ALL detections first
        for det in detections:
            box = det["box"]
            x1, y1, x2, y2 = int(max(0, box[0])), int(max(0, box[1])), int(min(w, box[2])), int(min(h, box[3]))
            body_crop = frame[y1:y2, x1:x2]
            
            if body_crop.size > 0:
                ch, cw = body_crop.shape[:2]
                resized = cv2.resize(body_crop, (256, 256))
                img_tensor = np.expand_dims(np.transpose(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), (2, 0, 1)), axis=0).astype(np.uint8)
                
                outs = self.face_net.run(None, {'image': img_tensor})
                out_names = [o.name for o in self.face_net.get_outputs()]
                box_coords_1 = outs[out_names.index('box_coords_1')]
                box_coords_2 = outs[out_names.index('box_coords_2')]
                box_scores_1 = outs[out_names.index('box_scores_1')]
                box_scores_2 = outs[out_names.index('box_scores_2')]
                
                scores_1 = (box_scores_1.astype(np.float32) - 255) * 12.9333
                scores_2 = (box_scores_2.astype(np.float32) - 246) * 0.3584
                scores = np.concatenate([scores_1.flatten(), scores_2.flatten()])
                
                best_idx = np.argmax(scores)
                if scores[best_idx] > -5.0:
                    coords_1 = (box_coords_1.astype(np.float32) - 192) * 1.7741
                    coords_2 = (box_coords_2.astype(np.float32) - 86) * 1.9781
                    coords = np.concatenate([coords_1.reshape(-1, 16), coords_2.reshape(-1, 16)])
                    
                    best_coords = coords[best_idx]
                    if best_idx < 512:
                        grid_y = (best_idx // 2) // 16
                        grid_x = (best_idx // 2) % 16
                        stride = 16
                    else:
                        idx = best_idx - 512
                        grid_y = (idx // 6) // 8
                        grid_x = (idx // 6) % 8
                        stride = 32
                        
                    anchor_x = (grid_x + 0.5) * stride
                    anchor_y = (grid_y + 0.5) * stride
                    
                    cx_256 = best_coords[0] + anchor_x
                    cy_256 = best_coords[1] + anchor_y
                    w_256 = best_coords[2]
                    h_256 = best_coords[3]
                    
                    cx = cx_256 * cw / 256.0
                    cy = cy_256 * ch / 256.0
                    fw = w_256 * cw / 256.0
                    fh = h_256 * ch / 256.0
                    
                    f_x1 = max(0, int(cx - fw * 0.75))
                    f_y1 = max(0, int(cy - fh * 0.75))
                    f_x2 = min(cw, int(cx + fw * 0.75))
                    f_y2 = min(ch, int(cy + fh * 0.75))
                    
                    face_roi = body_crop[f_y1:f_y2, f_x1:f_x2]
                    
                    if face_roi.size > 0:
                        f_h, f_w = face_roi.shape[:2]
                        resized_roi = cv2.resize(face_roi, (192, 192))
                        roi_tensor = np.expand_dims(np.transpose(cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB), (2, 0, 1)), axis=0).astype(np.uint8)
                        
                        lmk_outs = self.landmark_net.run(None, {'image': roi_tensor})
                        lmk_names = [o.name for o in self.landmark_net.get_outputs()]
                        landmarks_q = lmk_outs[lmk_names.index('landmarks')]
                        landmarks = (landmarks_q.astype(np.float32) - 50) * 0.004985
                        
                        r_eye_x = (landmarks[0, 33, 0] + landmarks[0, 133, 0]) / 2.0 * f_w + f_x1
                        r_eye_y = (landmarks[0, 33, 1] + landmarks[0, 133, 1]) / 2.0 * f_h + f_y1
                        l_eye_x = (landmarks[0, 362, 0] + landmarks[0, 263, 0]) / 2.0 * f_w + f_x1
                        l_eye_y = (landmarks[0, 362, 1] + landmarks[0, 263, 1]) / 2.0 * f_h + f_y1
                        
                        dY = r_eye_y - l_eye_y
                        dX = r_eye_x - l_eye_x
                        angle = np.degrees(np.arctan2(dY, dX)) - 180
                        
                        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
                        aligned_crop = cv2.warpAffine(body_crop, M, (cw, ch), flags=cv2.INTER_CUBIC)
                        
                        size = int(max(fw, fh) * 1.2)
                        half_size = size // 2
                        
                        crop_y1, crop_y2 = max(0, int(cy - half_size)), min(ch, int(cy + half_size))
                        crop_x1, crop_x2 = max(0, int(cx - half_size)), min(cw, int(cx + half_size))
                        
                        if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                            det["roi_frame"] = aligned_crop[crop_y1:crop_y2, crop_x1:crop_x2]

            is_pre_aligned = "roi_frame" in det
            embedding = self.inference_manager.execute_stage2_biometrics(
                det.get("roi_frame", frame[y1:y2, x1:x2]), 
                pre_aligned=is_pre_aligned
            )
            det["embedding"] = embedding

        # --- 2. REGISTRATION HOLD GUARD GATE (Moved after biometrics to secure real embeddings) ---
        is_database_empty = (len(self.profiles) == 0)
        
        with self.mutex:
            if is_database_empty:
                best_det = None
                best_score = float('-inf')
                for det in detections:
                    box = det["box"]
                    cx = (box[0] + box[2]) / 2.0
                    cy = (box[1] + box[3]) / 2.0
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    dist_to_center = math.hypot(cx - w/2, cy - h/2)
                    score = area - (dist_to_center * 0.5)
                    if score > best_score:
                        best_score = score
                        best_det = det
                
                if best_det is None:
                    return
                
                if self.primary_user_track_id not in self.tracked_persons:
                    track_hash = f"Person_hash_{int(current_time * 1000)}_{self.track_id_counter}"
                    self.track_id_counter += 1
                    primary_person = Person(track_hash, best_det["embedding"], best_det["box"])
                    self.tracked_persons[track_hash] = primary_person
                    self.primary_user_track_id = track_hash
                else:
                    primary_person = self.tracked_persons[self.primary_user_track_id]

                primary_person.name = "Unknown (Ready for Registration)"
                primary_person.state = "Awaiting Registration"
                primary_person.box = best_det["box"]
                primary_person.pose = best_det["pose"]
                primary_person.embedding = best_det["embedding"]
                primary_person.last_seen = current_time
                primary_person.last_update = current_time
                
                keys_to_delete = [k for k in self.tracked_persons.keys() if k != self.primary_user_track_id]
                for k in keys_to_delete:
                    del self.tracked_persons[k]

                if primary_person.state != getattr(primary_person, "last_state", "Unknown"):
                    print(f"[STATE CHANGE] {primary_person.name} transitioned from {getattr(primary_person, 'last_state', 'Unknown')} -> {primary_person.state}")
                    primary_person.last_state = primary_person.state
                    
                return

        # Phase B: Dynamic Identity Assignment (STRICT 3-PHASE EXECUTION MODEL)
        active_ids = set()
        phase1_biometric_matched_ids = set()
        unmatched_detections = []
        
        # ---------------------------------------------------------
        # PHASE 1: EXPLICIT BIOMETRIC BINDING
        # ---------------------------------------------------------
        for det in detections:
            box = det["box"]
            embedding = det["embedding"]
            validated_profile_name, calculated_similarity = self._match_profile(embedding, box=box, w=w)
            
            bound_person = None
            if validated_profile_name is not None:
                for track_id, p in self.tracked_persons.items():
                    if getattr(p, 'verification_status', 'UNKNOWN') == "VERIFIED" and getattr(p, 'verified_name', None) == validated_profile_name:
                        bound_person = p
                        break
            
            if bound_person is not None:
                bound_person.frame_val_name = validated_profile_name
                det["matched_person"] = bound_person
                active_ids.add(bound_person.track_id)
                phase1_biometric_matched_ids.add(bound_person.track_id)
            else:
                unmatched_detections.append((det, validated_profile_name, calculated_similarity))
                
        # ---------------------------------------------------------
        # PHASE 1.5: PARENT-CHILD CONTOUR SUPPRESSION (IoM Ghost Filter)
        # ---------------------------------------------------------
        filtered_unmatched_detections = []
        for det_tuple in unmatched_detections:
            det = det_tuple[0]
            box_a = det["box"]
            area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            
            is_contour_ghost = False
            for act_id in active_ids:
                p_box = self.tracked_persons[act_id].box
                area_b = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                
                inter_x1 = max(box_a[0], p_box[0])
                inter_y1 = max(box_a[1], p_box[1])
                inter_x2 = min(box_a[2], p_box[2])
                inter_y2 = min(box_a[3], p_box[3])
                
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                area_inter = inter_w * inter_h
                
                iom = area_inter / max(min(area_a, area_b), 1e-6)
                
                if iom > 0.55:
                    is_contour_ghost = True
                    break
                    
            if not is_contour_ghost:
                filtered_unmatched_detections.append(det_tuple)
                
        unmatched_detections = filtered_unmatched_detections
        
        # ---------------------------------------------------------
        # PHASE 2: FALLBACK SPATIAL ASSOCIATION (Scale-Invariant Centroid)
        # ---------------------------------------------------------
        remaining_tracks = [p for tid, p in self.tracked_persons.items() if tid not in active_ids]
        
        for det_tuple in unmatched_detections:
            det, val_name, calc_sim = det_tuple
            box = det["box"]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            
            best_cost = float('inf')
            best_guest = None
            
            for person in remaining_tracks:
                if person.track_id in active_ids:
                    continue
                p_box = person.box
                
                inter_x1 = max(box[0], p_box[0])
                inter_y1 = max(box[1], p_box[1])
                inter_x2 = min(box[2], p_box[2])
                inter_y2 = min(box[3], p_box[3])
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                
                box_area = (box[2] - box[0]) * (box[3] - box[1])
                p_box_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
                iou = inter_area / float(box_area + p_box_area - inter_area + 1e-6)
                
                p_cx = (p_box[0] + p_box[2]) / 2.0
                p_cy = (p_box[1] + p_box[3]) / 2.0
                
                dist = math.sqrt((cx - p_cx)**2 + (cy - p_cy)**2)
                p_width = max(p_box[2] - p_box[0], 1e-6)
                norm_dist = dist / p_width
                
                is_spatial_match = (iou > 0.45) or (norm_dist < 0.25)
                
                if is_spatial_match:
                    if norm_dist < best_cost:
                        best_cost = norm_dist
                        best_guest = person
                        
            if best_guest is not None:
                best_guest.frame_val_name = val_name
                det["matched_person"] = best_guest
                active_ids.add(best_guest.track_id)
                remaining_tracks.remove(best_guest)
            else:
                track_hash = f"Guest_hash_{int(current_time * 1000)}_{self.track_id_counter}"
                self.track_id_counter += 1
                best_guest = Person(track_hash, np.zeros(128, dtype=np.float32), box)
                best_guest.name = "Unknown"
                best_guest.state = "Unregistered Guest"
                best_guest.verification_status = "UNKNOWN"
                best_guest.biometric_match_counter = 0
                best_guest.candidate_name = None
                best_guest.verified_name = None
                best_guest.frame_val_name = val_name
                
                self.tracked_persons[track_hash] = best_guest
                det["matched_person"] = best_guest
                active_ids.add(best_guest.track_id)
                
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # PHASE 3: UNIFIED LIFECYCLE EVALUATION AND TEMPORAL TIME DECAY
        # ---------------------------------------------------------
        PROTECTED_STATES = {"Calibrating", "Posture Deficit Alert", "Session Limit Reached - Stand Up!", "Searching / Re-acquiring", "Absent", "Enrolling Multi-View"}
        for act_id in list(active_ids):
            person = self.tracked_persons[act_id]
            dt_step = current_time - person.last_update if person.last_update else 0.033
            
            if person.frame_val_name is not None:
                person.lost_grace_timer = 0.0
                
                if person.candidate_name == person.frame_val_name:
                    person.biometric_match_counter += 1
                else:
                    person.biometric_match_counter = 1
                    person.candidate_name = person.frame_val_name
                    
                if getattr(person, 'verification_status', 'UNKNOWN') != "VERIFIED":
                    person.verification_status = "VERIFYING"
                    person.name = "Verifying..."
                    person.state = "Verifying Identity"
                
                if person.biometric_match_counter >= 5:
                    name_conflict = False
                    for tid, p in self.tracked_persons.items():
                        if p.track_id != person.track_id and getattr(p, 'verification_status', 'UNKNOWN') == "VERIFIED" and getattr(p, 'verified_name', None) == person.frame_val_name:
                            name_conflict = True
                            break
                            
                    if not name_conflict:
                        person.verification_status = "VERIFIED"
                        person.verified_name = person.frame_val_name
                        person.name = person.frame_val_name
                        if person.state not in PROTECTED_STATES:
                            person.state = "Tracking Active"
                        self.primary_user_track_id = person.track_id
                        self.current_authenticated_user = person.frame_val_name
                        if hasattr(person, 'state_history_window'):
                            person.state_history_window.clear()
            else:
                person.biometric_match_counter = 0
                if getattr(person, 'verification_status', 'UNKNOWN') in ["VERIFIED", "VERIFYING"]:
                    person.lost_grace_timer += dt_step
                    if person.lost_grace_timer > 4.5:
                        print(f"[TRACKING] Grace window expired for {person.name}. Dropping state.")
                        person.state = "Searching / Re-acquiring"
                        person.name = "Unknown"
                        person.verification_status = "UNKNOWN"
                        person.biometric_match_counter = 0
                        person.candidate_name = None
                        person.verified_name = None
                    else:
                        if person.verification_status == "VERIFIED":
                            if person.state not in PROTECTED_STATES:
                                person.state = "Tracking Active"
                        elif person.verification_status == "VERIFYING":
                            person.state = "Verifying Identity"
                else:
                    if person.state != "Enrolling Multi-View":
                        person.verification_status = "UNKNOWN"
                        person.name = "Unknown"
                        person.state = "Unregistered Guest"

        # ---------------------------------------------------------
        # FINAL CLEANUP & POSTURE EVALUATION
        # ---------------------------------------------------------
        for det in detections:
            matched_person = det.get("matched_person")
            if not matched_person:
                continue
                
            if det is not None and matched_person.verification_status == "VERIFIED":
                if matched_person.state in ["Absent", "Searching / Re-acquiring", "Posture Deficit Alert"]:
                    matched_person.state = "Tracking Active"
                    matched_person.lost_grace_timer = 0.0
                
            box = det["box"]
            pose = det["pose"]
            shoulder_width = np.abs(pose["left_shoulder"].x - pose["right_shoulder"].x)
            
            matched_person.last_seen = current_time
            dt = current_time - matched_person.last_update if matched_person.last_update else 0.033
            matched_person.last_update = current_time
            matched_person.box = box
            matched_person.pose = pose
            if "embedding" in det:
                matched_person.embedding = det["embedding"]
                
                if matched_person.name == "Unknown" or matched_person.verification_status != "VERIFIED":
                    best_name, sim = self._match_profile(matched_person.embedding)
                    print(f"[RE-ACQUISITION SCAN] Evaluating live track... Best Match: {best_name} (Sim: {sim:.3f})")
                    
                    if not hasattr(matched_person, '_evidence_buffer'):
                        matched_person._evidence_buffer = []
                        
                    if best_name != "Unknown" and best_name is not None:
                        matched_person._evidence_buffer.append(best_name)
                    else:
                        matched_person._evidence_buffer.append("None")
                        
                    if len(matched_person._evidence_buffer) > 8:
                        matched_person._evidence_buffer.pop(0)
                        
                    if len(matched_person._evidence_buffer) == 8:
                        from collections import Counter
                        counts = Counter(matched_person._evidence_buffer)
                        top_candidate, count = counts.most_common(1)[0]
                        
                        if top_candidate != "None" and count >= 4:
                            matched_person.name = top_candidate
                            matched_person.verified_name = top_candidate
                            matched_person.verification_status = "VERIFIED"
                            matched_person.state = "Tracking Active"
                            matched_person.lost_grace_timer = 0.0
                            matched_person._evidence_buffer.clear()
                            print(f"[TRACKING] Temporal Consensus Achieved! Restored identity: {top_candidate}")
                        
                if matched_person.verification_status == "VERIFIED" and matched_person.state == "Tracking Active":
                    profile = self.profiles.get(matched_person.name)
                    if profile and "embeddings" in profile and len(profile["embeddings"]) < 15:
                        db_embs = profile["embeddings"]
                        norm_live = matched_person.embedding / (np.linalg.norm(matched_person.embedding) + 1e-6)
                        
                        max_sim = 0.0
                        for db_emb in db_embs:
                            db_emb_np = np.array(db_emb, dtype=np.float32)
                            norm_db = db_emb_np / (np.linalg.norm(db_emb_np) + 1e-6)
                            sim = np.dot(norm_live, norm_db)
                            if sim > max_sim: max_sim = sim
                            
                        if 0.78 <= max_sim <= 0.93:
                            profile["embeddings"].append(matched_person.embedding.tolist())
                            self._save_profiles_json()
                            print(f"[TRACKING] Background Cluster Accumulation: Captured new angle for {matched_person.name} (sim: {max_sim:.3f}). Total views: {len(profile['embeddings'])}")

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

            if matched_person.track_id == self.primary_user_track_id:
                gaze_x, is_looking_away = self.inference_manager.extract_pupil_gaze(frame, det["l_eye"], det["r_eye"])
                matched_person.gaze_x = gaze_x
                matched_person.is_looking_away = is_looking_away
                
                cached_status = getattr(matched_person, 'verification_status', 'UNKNOWN')
                cached_vname = getattr(matched_person, 'verified_name', None)
                cached_name = getattr(matched_person, 'name', 'Unknown')
                cached_timer = getattr(matched_person, 'lost_grace_timer', 0.0)
                
                self._evaluate_single_target_health(matched_person, pose, current_ratio, dt, current_time, frame_shape)
                
                matched_person.verification_status = cached_status
                matched_person.verified_name = cached_vname
                matched_person.name = cached_name
                matched_person.lost_grace_timer = cached_timer
                
            if matched_person.state != getattr(matched_person, "last_state", "Unknown"):
                print(f"[STATE CHANGE] {matched_person.name} transitioned from {getattr(matched_person, 'last_state', 'Unknown')} -> {matched_person.state}")
                matched_person.last_state = matched_person.state
                
            if current_time - getattr(matched_person, 'last_analytics_flush_time', 0.0) >= 10.0:
                self.db_manager.log_analytics_flush(
                    user_name=matched_person.name,
                    session_state=matched_person.state,
                    duration_seconds=10.0,
                    continuous_sitting_seconds=matched_person.sitting_duration_clock,
                    continuous_gaze_seconds=matched_person.screen_gaze_accumulation_timer,
                    average_head_pitch=getattr(matched_person, 'smoothed_pitch', matched_person.pitch),
                    ocular_break_accumulator=matched_person.ocular_break_timer
                )
                matched_person.last_analytics_flush_time = current_time

            # Reset frame validation for next loop
            matched_person.frame_val_name = None

        # --- 4. ASYMMETRIC STATE EVICTION ---
        for track_id, person in list(self.tracked_persons.items()):
            if track_id not in active_ids:
                if person.name and person.name not in ["Unknown / Bystander", "Unknown (Ready for Registration)", "Unknown"]:
                    if current_time - person.last_seen > 5.0:
                        if person.state != "Absent":
                            self.db_manager.log_session_metrics(person.name, "session_ended", person.sitting_duration_clock)
                            person.state = "Absent"
                            if hasattr(person, 'state_history_window'):
                                person.state_history_window.clear()
                    else:
                        if person.state not in ["Calibrating", "Absent"]:
                            person.state = "Searching / Re-acquiring"
                else:
                    del self.tracked_persons[track_id]
                    
                if track_id in self.tracked_persons:
                    if person.state != getattr(person, "last_state", "Unknown"):
                        print(f"[STATE CHANGE] {person.name} transitioned from {getattr(person, 'last_state', 'Unknown')} -> {person.state}")
                        person.last_state = person.state
