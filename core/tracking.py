import time
import threading
import numpy as np

class Target:
    def __init__(self, track_id, embedding, box):
        self.track_id = track_id
        self.embedding = embedding
        self.box = box
        self.name = "Unknown"
        self.state = "Unregistered Guest - Monitoring Suspended"
        
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
        self.biometric_cutoff = 0.35
        
        self.baseline_pitch = 0.0
        self.baseline_shoulder_ratio = 0.0
        self.calibration_start = 0.0
        self.calibration_pitch_samples = []
        self.calibration_shoulder_samples = []
        
        self.smoothed_pitch = None
        self.smoothed_ratio = None

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

    def _match_profile(self, embedding):
        best_match = None
        best_dist = float('inf')
        for name, profile in self.profiles.items():
            db_emb = profile["embedding"]
            dist = np.linalg.norm(embedding - db_emb)
            cutoff = profile.get("biometric_cutoff", 0.35)
            if dist <= cutoff and dist < best_dist:
                best_dist = dist
                best_match = name
        return best_match

    def update(self, frame, detections, frame_shape):
        return self.process_frame_mot(frame, detections, frame_shape)

    def process_frame_mot(self, frame, detections, frame_shape):
        current_time = time.time()
        h, w = frame_shape[:2]
        active_ids = set()
        
        # Primary User Selection Gate:
        # Rank valid skeletons based on proximity score combining maximum shoulder width
        # (closest to the camera) and minimum horizontal distance to center of the frame screen.
        ranked_detections = []
        for det in detections:
            box = det["box"]
            shoulders = det["shoulders"]
            centroid_x = (box[0] + box[2]) / 2.0
            shoulder_width = np.abs(shoulders[1][0] - shoulders[0][0])
            
            # Combine max shoulder width (near camera) and min distance to centerline
            score = (shoulder_width / w) - (np.abs(centroid_x - w/2) / w)
            ranked_detections.append((score, det))
            
        # Sort descending by proximity score
        ranked_detections.sort(key=lambda x: x[0], reverse=True)
        
        with self.mutex:
            primary_det = None
            if ranked_detections:
                primary_det = ranked_detections[0][1]
                
            for score, det in ranked_detections:
                embedding = det["embedding"]
                box = det["box"]
                shoulders = det["shoulders"]
                nose = det["nose"]
                det_centroid = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                shoulder_width = np.abs(shoulders[1][0] - shoulders[0][0])
                
                is_primary = (det is primary_det)
                
                matched_target = None
                best_cost = float('inf')
                
                for tid, target in self.targets.items():
                    tc = target.get_centroid()
                    centroid_dist = np.linalg.norm(np.array(det_centroid) - np.array(tc))
                    scale_aware_dist = centroid_dist / (shoulder_width + 1e-6)
                    emb_dist = np.linalg.norm(embedding - target.embedding)
                    
                    cost = (scale_aware_dist * 0.5) + (emb_dist * 0.5)
                    if cost <= 0.35 and cost < best_cost:
                        best_cost = cost
                        matched_target = target
                        
                if matched_target is None:
                    track_hash = f"Person_hash_{int(current_time * 1000)}_{self.track_counter}"
                    self.track_counter += 1
                    matched_target = Target(track_hash, embedding, box)
                    self.targets[track_hash] = matched_target

                dt = current_time - matched_target.last_update
                matched_target.last_seen = current_time
                matched_target.last_update = current_time
                matched_target.box = box
                active_ids.add(matched_target.track_id)
                
                matched_target.pitch = det["pitch"]
                matched_target.yaw = det["yaw"]
                matched_target.roll = det["roll"]
                
                shoulder_y = (shoulders[0][1] + shoulders[1][1]) / 2.0
                nose_y = nose[1]
                current_ratio = (shoulder_y - nose_y) / (shoulder_width + 1e-6)
                
                if matched_target.smoothed_pitch is None:
                    matched_target.smoothed_pitch = det["pitch"]
                    matched_target.smoothed_ratio = current_ratio
                else:
                    alpha = 0.15
                    matched_target.smoothed_pitch = (1 - alpha) * matched_target.smoothed_pitch + alpha * det["pitch"]
                    matched_target.smoothed_ratio = (1 - alpha) * matched_target.smoothed_ratio + alpha * current_ratio
                
                if not is_primary:
                    # Secondary background tracks or walking targets must be designated as 'Secondary Bystander'.
                    # For any track marked as a bystander, completely freeze their internal clocks,
                    # bypass database lookup validations, and do NOT render individual state metrics cards.
                    matched_target.state = "Secondary Bystander"
                    continue
                
                # Primary target posture & calibration evaluation
                if matched_target.name == "Unknown":
                    profile_name = self._match_profile(embedding)
                    if profile_name:
                        matched_target.name = profile_name
                        p = self.profiles[profile_name]
                        matched_target.slouch_sensitivity = p["slouch_sensitivity"]
                        matched_target.session_limit = p["session_limit"]
                        matched_target.stand_requirement = p["stand_requirement"]
                        matched_target.gaze_away_limit = p["gaze_away_limit"]
                        
                        matched_target.state = "Calibrating"
                        matched_target.calibration_start = current_time
                        matched_target.calibration_pitch_samples = []
                        matched_target.calibration_shoulder_samples = []
                    else:
                        matched_target.state = "Unregistered Guest - Monitoring Suspended"
                        continue

                if matched_target.name == "Unknown":
                    matched_target.state = "Unregistered Guest - Monitoring Suspended"
                    continue
                    
                gaze_x, is_looking_away = self.inference_engine.extract_pupil_gaze(frame, det["l_eye"], det["r_eye"])
                matched_target.gaze_x = gaze_x
                matched_target.is_looking_away = is_looking_away
                
                if matched_target.state == "Calibrating":
                    elapsed_calib = current_time - matched_target.calibration_start
                    matched_target.calibration_pitch_samples.append(matched_target.pitch)
                    matched_target.calibration_shoulder_samples.append(current_ratio)
                    
                    if elapsed_calib >= 3.0:
                        matched_target.baseline_pitch = float(np.mean(matched_target.calibration_pitch_samples))
                        matched_target.baseline_shoulder_ratio = float(np.mean(matched_target.calibration_shoulder_samples))
                        matched_target.state = "Tracking Active"
                    continue
                
                # Camera-Agnostic Posture Evaluation
                relative_slouch = matched_target.pitch - matched_target.baseline_pitch
                
                # Proportional Standing Gate
                is_standing = current_ratio < (matched_target.baseline_shoulder_ratio * 0.82)
                matched_target.is_standing = is_standing

                if matched_target.is_looking_away:
                    matched_target.gaze_away_clock += dt
                    if matched_target.gaze_away_clock > matched_target.gaze_away_limit:
                        matched_target.state = "Ocular Break Recommended"
                    else:
                        matched_target.state = "Looking Away"
                    continue
                else:
                    matched_target.gaze_away_clock = 0.0

                if matched_target.is_standing:
                    matched_target.state = "Standing"
                    matched_target.standing_duration_clock += dt
                    if matched_target.standing_duration_clock >= matched_target.stand_requirement:
                        matched_target.sitting_duration_clock = 0.0
                else:
                    matched_target.sitting_duration_clock += dt
                    if matched_target.sitting_duration_clock >= matched_target.session_limit:
                        matched_target.state = "Session Limit Reached - Stand Up!"
                        matched_target.standing_duration_clock = 0.0
                    else:
                        if relative_slouch > matched_target.slouch_sensitivity:
                            matched_target.state = "Posture Deficit Alert"
                            matched_target.slouch_timer += dt
                        else:
                            matched_target.state = "Tracking Active"
                            matched_target.slouch_timer = 0.0
                                    
            to_delete = []
            for tid, target in list(self.targets.items()):
                if tid not in active_ids:
                    # Freeze internal clocks if missing
                    elapsed = current_time - target.last_seen
                    if elapsed > 10.0:
                        to_delete.append(tid)
                        if target.name != "Unknown":
                            self.db_manager.log_session_metrics(target.name, "session_ended", target.sitting_duration_clock)
                    else:
                        if target.state != "Calibrating" and target.state != "Secondary Bystander":
                            target.state = "Coasting (Paused)"
                    
            for tid in to_delete:
                del self.targets[tid]
