import time
import threading
import numpy as np
import multiprocessing
from typing import Any, Optional, Tuple
from core.biometrics import CrossPlatformInferenceManager
from .state import Person, VoiceAlertDaemon
from .geometry import compute_ious, compute_normalized_distances, filter_overlapping_detections

def bg_biometric_process_worker(job_queue, result_queue, profiles):
    import cv2
    import numpy as np
    import os
    import onnxruntime as ort
    from core.biometrics import CrossPlatformInferenceManager

    inference_manager = CrossPlatformInferenceManager()
    base_dir = "D:\\Thundersoft\\dbot"
    model_path = os.path.join(base_dir, "models", "face_detector.onnx")
    landmark_path = os.path.join(base_dir, "models", "face_landmark_detector.onnx")
    face_net = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    landmark_net = ort.InferenceSession(landmark_path, providers=['CPUExecutionProvider'])

    while True:
        try:
            job = job_queue.get()
            if job is None:
                continue
            
            track_id = job["track_id"]
            frame = job["frame"]
            box = job["box"]
            w, h = job["w"], job["h"]
            
            x1, y1, x2, y2 = int(max(0, box[0])), int(max(0, box[1])), int(min(w, box[2])), int(min(h, box[3]))
            body_crop = frame[y1:y2, x1:x2]
            embedding = None
            
            if body_crop.size > 0:
                ch, cw = body_crop.shape[:2]
                resized = cv2.resize(body_crop, (256, 256))
                img_tensor = np.expand_dims(np.transpose(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), (2, 0, 1)), axis=0).astype(np.uint8)
                
                outs = face_net.run(None, {'image': img_tensor})
                out_names = [o.name for o in face_net.get_outputs()]
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
                    
                    det_info = {}
                    if face_roi.size > 0:
                        f_h, f_w = face_roi.shape[:2]
                        resized_roi = cv2.resize(face_roi, (192, 192))
                        roi_tensor = np.expand_dims(np.transpose(cv2.cvtColor(resized_roi, cv2.COLOR_BGR2RGB), (2, 0, 1)), axis=0).astype(np.uint8)
                        
                        lmk_outs = landmark_net.run(None, {'image': roi_tensor})
                        lmk_names = [o.name for o in landmark_net.get_outputs()]
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
                            det_info["roi_frame"] = aligned_crop[crop_y1:crop_y2, crop_x1:crop_x2]
                    
                    is_pre_aligned = "roi_frame" in det_info
                    embedding = inference_manager.execute_stage2_biometrics(
                        det_info.get("roi_frame", frame[y1:y2, x1:x2]), 
                        pre_aligned=is_pre_aligned
                    )
            
            if embedding is not None:
                matched_db_profile_string = None
                highest_consensus = -1.0
                second_highest_consensus = -1.0
                norm_embedding = embedding / (np.linalg.norm(embedding) + 1e-6)
                
                for name, profile in profiles.items():
                    db_embs = profile.get("embeddings", [])
                    if "embedding" in profile and len(db_embs) == 0:
                        db_embs = [profile["embedding"]]
                        
                    if len(db_embs) == 0:
                        continue
                        
                    cluster_matrix = np.array([np.array(e, dtype=np.float32) for e in db_embs])
                    cluster_matrix_unit = cluster_matrix / (np.linalg.norm(cluster_matrix, axis=1, keepdims=True) + 1e-6)
                    
                    for db_emb_unit in cluster_matrix_unit:
                        cos_sim = np.dot(norm_embedding, db_emb_unit)
                        l2_dist = np.linalg.norm(norm_embedding - db_emb_unit)
                        
                        if cos_sim >= 0.84 and l2_dist < 0.58:
                            if cos_sim > highest_consensus:
                                second_highest_consensus = highest_consensus
                                highest_consensus = cos_sim
                                matched_db_profile_string = name
                            elif cos_sim > second_highest_consensus:
                                second_highest_consensus = cos_sim
                                
                if len(profiles) == 0:
                    print("[MULTIPROCESS WORKER] Database empty. Operating in Registration Bootstrap mode.")
                    highest_consensus = 1.0
                    margin = 1.0
                    validated_profile_name = "Registration_Candidate"
                    
                    result_queue.put_nowait({
                        "track_id": track_id,
                        "name": validated_profile_name,
                        "sim": highest_consensus,
                        "margin": margin,
                        "embedding": embedding
                    })
                else:
                    if highest_consensus >= 0.84:
                        validated_profile_name = matched_db_profile_string
                    else:
                        validated_profile_name = "Unknown"
                        
                    if validated_profile_name is not None and validated_profile_name != "Unknown":
                        # Fix for single-profile workspaces to prevent self-variance margin failure
                        if len(profiles) == 1:
                            second_highest_consensus = 0.0
                            
                        margin = highest_consensus - max(second_highest_consensus, 0.0)
                        print(f"[MULTIPROCESS WORKER] Identity matched: {validated_profile_name} (Sim: {highest_consensus:.3f}, Margin: {margin:.3f})")
                        result_queue.put_nowait({
                            "track_id": track_id,
                            "name": validated_profile_name,
                            "sim": highest_consensus,
                            "margin": margin,
                            "embedding": embedding
                        })
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[MULTIPROCESS WORKER ERROR] {e}")

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
        
        # Multiprocessing Biometric Engine
        self.job_queue = multiprocessing.Queue(maxsize=1)
        self.result_queue = multiprocessing.Queue()
        self.bg_process = multiprocessing.Process(target=bg_biometric_process_worker, args=(self.job_queue, self.result_queue, self.profiles), daemon=True)
        self.bg_process.start()
        
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
                
            cluster_matrix = np.array([np.array(e, dtype=np.float32) for e in db_embs])
            cluster_matrix_unit = cluster_matrix / (np.linalg.norm(cluster_matrix, axis=1, keepdims=True) + 1e-6)
            
            for db_emb_unit in cluster_matrix_unit:
                cos_sim = np.dot(norm_embedding, db_emb_unit)
                l2_dist = np.linalg.norm(norm_embedding - db_emb_unit)
                
                if cos_sim >= 0.84 and l2_dist < 0.58:
                    if cos_sim > highest_consensus:
                        highest_consensus = cos_sim
                        matched_db_profile_string = name
                        
        if highest_consensus >= 0.84:
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
            person.next_state = "Unregistered Target"
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
                person.next_state = "Tracking Active"
                        
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
                person.next_state = "Standing"
            elif person.slouch_accumulator_time >= 4.0:
                person.next_state = "Posture Deficit Alert"
            elif person.active_accumulator_time >= 1.5:
                if person.state in ["Standing", "Posture Deficit Alert"]:
                    person.next_state = "Tracking Active"

            if person.screen_gaze_accumulation_timer >= person.screen_gaze_limit:
                person.next_state = "Ocular Break Recommended"
                if not person.ocular_break_announced:
                    print(f"[TIMER ALERT] Triggering Voice Alert for: {person.state}")
                    self._dispatch_voice(f"{person.name}, attention, eye strain warning. Please look away from the screen.")
                    person.ocular_break_announced = True
                
                if person.is_looking_away:
                    person.ocular_break_timer += dt
                    if person.ocular_break_timer >= person.gaze_away_limit:
                        person.screen_gaze_accumulation_timer = 0.0
                        person.ocular_break_timer = 0.0
                        person.next_state = "Tracking Active"
                        person.ocular_break_announced = False

            if person.state in ["Tracking Active", "Posture Deficit Alert", "Ocular Break Recommended"]:
                person.sitting_duration_clock += dt
                if person.sitting_duration_clock >= person.session_limit:
                    person.next_state = "Session Limit Reached - Stand Up!"
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


    def process_frame_mot(self, frame, frame_shape):
        """
        Executes multi-object tracking and health evaluations on each incoming frame.
        """
        import time
        import numpy as np
        import queue as python_queue
        
        best_guest = None
        current_time = time.time()
        
        # ---------------------------------------------------------
        # 1. STATE INGESTION (QUEUE DRAIN)
        # ---------------------------------------------------------
        while not self.result_queue.empty():
            try:
                res = self.result_queue.get_nowait()
                track_id = res["track_id"]
                with self.mutex:
                    if track_id in self.tracked_persons:
                        p = self.tracked_persons[track_id]
                        if "embedding" in res:
                            p.embedding = res["embedding"]
                        sim = res.get("sim", 0.0)
                        margin = res.get("margin", 0.0)
                        candidate_name = res["name"]
                        
                        current_area = (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])
                        min_primary_area = (frame_shape[0] * frame_shape[1]) * 0.15
                        
                        if candidate_name == "Meghan" and current_area < min_primary_area:
                            print(f"[SECURITY REJECTION] Blocked background track {track_id} from claiming primary identity due to Absolute Workspace Boundary Violation.")
                            p.name = "Unknown"
                            p.state = "Unregistered Guest"
                            p.verification_status = "UNKNOWN"
                            p.biometric_match_counter = 0
                            continue
                        
                        sim_threshold = 0.910 if hasattr(self, 'profiles') and len(self.profiles) == 1 else 0.88
                        if sim >= sim_threshold and margin > 0.05 and candidate_name != "Unknown":
                            p.candidate_name = candidate_name
                            p.biometric_match_counter = getattr(p, 'biometric_match_counter', 0) + 1
                            if p.biometric_match_counter >= 5:
                                conflict_track = None
                                for other_track_id, other_p in self.tracked_persons.items():
                                    if other_track_id != p.track_id and getattr(other_p, 'verification_status', 'UNKNOWN') == "VERIFIED" and other_p.state == "Tracking Active" and other_p.name == p.candidate_name:
                                        conflict_track = other_p
                                        break
                                
                                assign_identity = True
                                if conflict_track:
                                    p_area = (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])
                                    conflict_area = (conflict_track.box[2] - conflict_track.box[0]) * (conflict_track.box[3] - conflict_track.box[1])
                                    
                                    if p_area > (conflict_area * 1.3):
                                        print(f"[SECURITY] Identity Conflict Resolved: Foreground track ({p_area:.0f}) stole '{p.candidate_name}' from background ({conflict_area:.0f})")
                                        conflict_track.name = "Unknown"
                                        conflict_track.state = "Unregistered Guest"
                                        conflict_track.next_state = "Unregistered Guest"
                                        conflict_track.verification_status = "UNKNOWN"
                                        conflict_track.biometric_match_counter = 0
                                    else:
                                        print(f"[SECURITY] Identity Conflict Resolved: Rejected background track attempt to hijack '{p.candidate_name}'")
                                        assign_identity = False
                                        p.name = "Unknown"
                                        p.state = "Unregistered Guest"
                                        p.next_state = "Unregistered Guest"
                                        p.verification_status = "UNKNOWN"
                                        p.biometric_match_counter = 0

                                if assign_identity:
                                    p.verification_status = "VERIFIED"
                                    p.name = p.candidate_name
                                    p.verified_name = p.candidate_name
                                    p.frame_val_name = p.candidate_name
                                    p.next_state = "Tracking Active"
                                    
                                    if getattr(p, 'state', None) != "Tracking Active":
                                        print(f"[STATE CHANGE] {p.name} transitioned from {p.state} -> Tracking Active")
                                        p.state = "Tracking Active"
                                    p.last_state = p.state
                                    
                                    # Set tracking lock if meghan
                                    if p.name == "Meghan":
                                        self.primary_user_track_id = p.track_id
                                        self.current_authenticated_user = p.name
                            else:
                                p.next_state = "Candidate"
                                if getattr(p, 'state', None) != "Candidate":
                                    print(f"[STATE CHANGE] Track {p.track_id} transitioned from {p.state} -> Candidate ({p.biometric_match_counter}/5)")
                                    p.state = "Candidate"
                                p.last_state = p.state
                        else:
                            # Instant safety reset on a bad or suspicious frame
                            p.biometric_match_counter = 0
                            p.next_state = "Searching / Re-acquiring"
                            if getattr(p, 'state', None) != "Searching / Re-acquiring":
                                print(f"[STATE CHANGE] Track {p.track_id} transitioned from {p.state} -> Searching / Re-acquiring (Quality Gate Failed)")
                                p.state = "Searching / Re-acquiring"
                            p.last_state = p.state
            except python_queue.Empty:
                break
        
        # Handle registration and manual recalibration events cleanly...
        with self.mutex:
            if getattr(self, 'pending_registration_name', None):
                target_person = None
                if self.primary_user_track_id in self.tracked_persons:
                    target_person = self.tracked_persons[self.primary_user_track_id]
                else:
                    for person in self.tracked_persons.values():
                        if getattr(person, 'verification_status', 'UNKNOWN') == "VERIFIED":
                            target_person = person
                            break
                        if person.name == "Unknown (Ready for Registration)":
                            target_person = person
                            break
                            
                if target_person:
                    try:
                        face_embedding = target_person.embedding
                        
                        import numpy as np
                        emb_array = np.array(face_embedding)
                        if np.all(emb_array == 0.0) or emb_array.size == 0:
                            print("[TRACKING WARNING] Aborting save: Face embedding is an unpopulated zero-array placeholder.")
                            self.pending_registration_name = None
                        else:
                            # Check what save method actually exists on your database manager instance
                            if hasattr(self.db_manager, 'create_profile'):
                                self.db_manager.create_profile(self.pending_registration_name, face_embedding)
                            elif hasattr(self.db_manager, 'register_user'):
                                self.db_manager.register_user(self.pending_registration_name, face_embedding)
                            elif hasattr(self.db_manager, 'save_profile'):
                                self.db_manager.save_profile(self.pending_registration_name, face_embedding)
                            elif hasattr(self.db_manager, 'add_profile'):
                                self.db_manager.add_profile(self.pending_registration_name, face_embedding)
                            else:
                                # Fallback direct serialization if no matching abstraction method is found
                                import json
                                import os
                                cache_data = {"profiles": {self.pending_registration_name: face_embedding.tolist() if hasattr(face_embedding, "tolist") else face_embedding}}
                                if os.path.exists('profiles_cache.json'):
                                    try:
                                        with open('profiles_cache.json', 'r') as f:
                                            existing_data = json.load(f)
                                            if "profiles" in existing_data:
                                                existing_data["profiles"].update(cache_data["profiles"])
                                            cache_data = existing_data
                                    except Exception:
                                        pass
                                with open('profiles_cache.json', 'w') as f:
                                    json.dump(cache_data, f)
                                    
                            self.profiles[self.pending_registration_name] = {"embeddings": [face_embedding.tolist() if hasattr(face_embedding, "tolist") else face_embedding]}
                            
                            target_person.name = self.pending_registration_name
                            target_person.verification_status = "VERIFIED"
                            target_person.verified_name = self.pending_registration_name
                            target_person.next_state = "Tracking Active"
                            target_person.state = "Tracking Active"
                            
                            self.primary_user_track_id = target_person.track_id
                            self.current_authenticated_user = target_person.name
                            print(f"[TRACKING] Instant registration successful for {self.pending_registration_name}")
                            self._save_profiles_json()
                    except Exception as e:
                        print(f"[TRACKING] Failed instant registration: {e}")
                self.pending_registration_name = None
                
            if getattr(self, 'trigger_recalibration', False) or getattr(self, 'manual_recalibration_requested', False):
                if self.primary_user_track_id in self.tracked_persons:
                    person = self.tracked_persons[self.primary_user_track_id]
                    person.calibration_start_time = time.time()
                    person.calibration_accumulator = []
                    person.calibration_pitch_acc = []
                    person.calibration_y_acc = []
                    person.next_state = "Calibrating"
                    person.state = "Calibrating"
                    person.is_posture_calibrated = False
                    person.calibration_announced = False
                    print(f"[TRACKING] Safely triggered recalibration for {person.name}")
                self.trigger_recalibration = False
                self.manual_recalibration_requested = False

        # ---------------------------------------------------------
        # 2. DETECTION & SPATIAL ASSIGNMENT (FAST PATH)
        # ---------------------------------------------------------
        detections = self.inference_manager.execute_stage1_detector(frame)
        active_track_ids = set()
        
        if detections is not None and len(detections) > 0:
            verified_tracks = [p for p in self.tracked_persons.values() if p.verification_status == "VERIFIED" or p.name == "Meghan"]
            detections = filter_overlapping_detections(detections, verified_tracks)

            remaining_tracks = list(self.tracked_persons.values())
            
            if len(detections) > 0:
                if len(remaining_tracks) > 0:
                    det_boxes = np.array([d["box"] for d in detections])
                    trk_boxes = np.array([p.box for p in remaining_tracks])
                    
                    ious = compute_ious(det_boxes, trk_boxes)
                    norm_dists = compute_normalized_distances(det_boxes, trk_boxes)
                else:
                    ious = None
                    norm_dists = None
                
                for i, det in enumerate(detections):
                    box = det["box"]
                    best_guest = None
                    mot_id = det.get("track_id")
                    
                    if mot_id is not None and mot_id in self.tracked_persons:
                        best_guest = self.tracked_persons[mot_id]
                    else:
                        best_cost = float('inf')
                        if len(remaining_tracks) > 0:
                            for j, person in enumerate(remaining_tracks):
                                if person.track_id in active_track_ids:
                                    continue
                                
                                iou = ious[i, j]
                                norm_dist = norm_dists[i, j]
                                
                                if iou > 0.45 or norm_dist < 0.25:
                                    if norm_dist < best_cost:
                                        best_cost = norm_dist
                                        best_guest = person
                                        
                    if best_guest is not None:
                        det["matched_person"] = best_guest
                        active_track_ids.add(best_guest.track_id)
                    else:
                        track_hash = mot_id if mot_id is not None else f"Guest_hash_{int(current_time * 1000)}_{self.track_id_counter}"
                        self.track_id_counter += 1
                        best_guest = Person(track_hash, np.zeros(128, dtype=np.float32), box)
                        best_guest.name = "Unknown"
                        best_guest.state = "Unregistered Guest"
                        best_guest.next_state = "Unregistered Guest"
                        best_guest.verification_status = "UNKNOWN"
                        self.tracked_persons[track_hash] = best_guest
                        det["matched_person"] = best_guest
                        active_track_ids.add(best_guest.track_id)
                    
                    # Update Person with Detection
                    best_guest.last_seen = current_time
                    dt = current_time - best_guest.last_update if best_guest.last_update else 0.033
                    best_guest.last_update = current_time
                    best_guest.box = box
                    best_guest.pose = det.get("pose", getattr(best_guest, 'pose', None))
                    best_guest.pitch = (best_guest.pitch * 0.6) + (det.get("pitch", best_guest.pitch) * 0.4)
                    best_guest.yaw = det.get("yaw", best_guest.yaw)
                    best_guest.roll = det.get("roll", best_guest.roll)
                    
                    pose = best_guest.pose
                    if pose:
                        shoulder_width = np.abs(pose["left_shoulder"].x - pose["right_shoulder"].x)
                        shoulder_center_y = (pose["left_shoulder"].y + pose["right_shoulder"].y) / 2.0
                        current_ratio = np.abs(shoulder_center_y - pose["nose"].y) / max(shoulder_width, 1e-6)
                        best_guest.last_y = shoulder_center_y
                        
                        alpha = 0.10
                        if best_guest.smoothed_pitch is None:
                            best_guest.smoothed_pitch = best_guest.pitch
                            best_guest.smoothed_ratio = current_ratio
                            best_guest.smoothed_y = shoulder_center_y
                        else:
                            best_guest.smoothed_pitch = (1 - alpha) * best_guest.smoothed_pitch + alpha * best_guest.pitch
                            best_guest.smoothed_ratio = (1 - alpha) * best_guest.smoothed_ratio + alpha * current_ratio
                            best_guest.smoothed_y = (1 - alpha) * best_guest.smoothed_y + alpha * shoulder_center_y

                        if best_guest.track_id == self.primary_user_track_id and "l_eye" in det and "r_eye" in det:
                            gaze_x, is_looking_away = self.inference_manager.extract_pupil_gaze(frame, det["l_eye"], det["r_eye"])
                            best_guest.gaze_x = gaze_x
                            best_guest.is_looking_away = is_looking_away
                            self._evaluate_single_target_health(best_guest, pose, current_ratio, dt, current_time, frame_shape)
                    
                    # Background Cluster Accumulation
                    if best_guest.verification_status == "VERIFIED" and best_guest.state == "Tracking Active":
                        profile = self.profiles.get(best_guest.name)
                        if profile and "embeddings" in profile and len(profile["embeddings"]) < 15 and hasattr(best_guest, 'embedding') and np.sum(best_guest.embedding) != 0:
                            db_embs = profile["embeddings"]
                            norm_live = best_guest.embedding / (np.linalg.norm(best_guest.embedding) + 1e-6)
                            max_sim = 0.0
                            for db_emb in db_embs:
                                db_emb_np = np.array(db_emb, dtype=np.float32)
                                norm_db = db_emb_np / (np.linalg.norm(db_emb_np) + 1e-6)
                                sim = np.dot(norm_live, norm_db)
                                if sim > max_sim: max_sim = sim
                            if 0.78 <= max_sim <= 0.93:
                                profile["embeddings"].append(best_guest.embedding.tolist())
                                self._save_profiles_json()
                                print(f"[TRACKING] Background Cluster Accumulation: Captured new angle for {best_guest.name} (sim: {max_sim:.3f}). Total views: {len(profile['embeddings'])}")

                    # Analytics Flush
                    if current_time - getattr(best_guest, 'last_analytics_flush_time', 0.0) >= 10.0:
                        self.db_manager.log_analytics_flush(
                            user_name=best_guest.name,
                            session_state=best_guest.state,
                            duration_seconds=10.0,
                            continuous_sitting_seconds=best_guest.sitting_duration_clock,
                            continuous_gaze_seconds=best_guest.screen_gaze_accumulation_timer,
                            average_head_pitch=getattr(best_guest, 'smoothed_pitch', best_guest.pitch),
                            ocular_break_accumulator=best_guest.ocular_break_timer
                        )
                        best_guest.last_analytics_flush_time = current_time

        # ---------------------------------------------------------
        # 2b. PRIMARY USER PROXIMITY GUARD (ANTI-HIJACKING)
        # ---------------------------------------------------------
        active_tracks = [p for p in self.tracked_persons.values() if p.track_id in active_track_ids]
        verified_tracks = [p for p in active_tracks if p.verification_status == "VERIFIED" and p.state == "Tracking Active"]
        unverified_tracks = [p for p in active_tracks if getattr(p, 'verification_status', 'UNKNOWN') != "VERIFIED"]
        
        for v_track in verified_tracks:
            v_area = (v_track.box[2] - v_track.box[0]) * (v_track.box[3] - v_track.box[1])
            for u_track in unverified_tracks:
                u_area = (u_track.box[2] - u_track.box[0]) * (u_track.box[3] - u_track.box[1])
                if u_area > (v_area * 1.5):
                    print(f"[SECURITY] Proximity Guard Triggered! Unverified foreground track is suppressing background verified track '{v_track.name}'")
                    v_track.state = "Unregistered Guest"
                    v_track.next_state = "Unregistered Guest"
                    v_track.name = "Unknown"
                    v_track.verification_status = "UNKNOWN"
                    v_track.biometric_match_counter = 0
                    
                    u_track.biometric_match_counter = 0
                    u_track.state = "Verifying Identity"
                    u_track.next_state = "Verifying Identity"
                    
                    if getattr(self, 'primary_user_track_id', None) == v_track.track_id:
                        self.primary_user_track_id = None
                        self.current_authenticated_user = None
                    break

        # ---------------------------------------------------------
        # 3. BIOMETRIC OFFLOAD (ASYNC SUBMISSION)
        # ---------------------------------------------------------
        if (not hasattr(self, 'profiles') or len(self.profiles) == 0) and getattr(self, 'pending_registration_name', None) is None:
            print("[TRACKING INITIALIZER] Clean database footprint caught. Autoloading calibration target: 'Meghan'")
            self.pending_registration_name = "Meghan"
        is_profile_already_active = any(
            getattr(p, 'verification_status', 'UNKNOWN') == "VERIFIED" and p.state == "Tracking Active"
            for p in self.tracked_persons.values()
        )
        
        if self.job_queue.empty() and not is_profile_already_active:
            for act_id in active_track_ids:
                person = self.tracked_persons[act_id]
                if getattr(person, 'verification_status', 'UNKNOWN') != "VERIFIED":
                    try:
                        self.job_queue.put_nowait({
                            "track_id": person.track_id,
                            "frame": frame.copy(),
                            "box": person.box,
                            "w": frame_shape[1],
                            "h": frame_shape[0]
                        })
                        person.next_state = "Verifying Identity"
                        person.verification_status = "VERIFYING"
                        break # Only queue one job at a time
                    except python_queue.Full:
                        pass
                        
        # ---------------------------------------------------------
        # 4. TEMPORAL LIFECYCLE & STATE EVICTION
        # ---------------------------------------------------------
        keys_to_delete = []
        for track_id, person in list(self.tracked_persons.items()):
            if track_id not in active_track_ids:
                time_since_seen = current_time - person.last_seen
                if time_since_seen > 5.0:
                    if person.name and person.name not in ["Unknown / Bystander", "Unknown (Ready for Registration)", "Unknown"]:
                        self.db_manager.log_session_metrics(person.name, "session_ended", person.sitting_duration_clock)
                    keys_to_delete.append(track_id)
                elif time_since_seen > 2.0:
                    if person.state not in ["Calibrating", "Absent"]:
                        person.next_state = "Searching / Re-acquiring"
            else:
                if person.verification_status == "VERIFIED":
                    if person.state in ["Absent", "Searching / Re-acquiring", "Posture Deficit Alert"]:
                        person.next_state = "Tracking Active"
                        person.lost_grace_timer = 0.0
                elif person.verification_status == "UNKNOWN":
                    person.next_state = "Unregistered Guest"
            
            # State Emitting
            if hasattr(person, 'next_state') and person.next_state != person.state:
                if person.name is not None and person.name != "Unknown":
                    print(f"[STATE CHANGE] {person.name} transitioned from {person.state} -> {person.next_state}")
                person.state = person.next_state
                person.last_state = person.state
            
        for k in keys_to_delete:
            del self.tracked_persons[k]

