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
        
        # Calibration State
        self.baseline_pitch = 0.0
        self.baseline_shoulder_y = 0.0
        self.calibration_timer = 0.0
        self.calibration_start = 0.0
        self.calibration_pitch_samples = []
        self.calibration_shoulder_samples = []

    def get_centroid(self):
        return ((self.box[0] + self.box[2])/2, (self.box[1] + self.box[3])/2)

class TrackerEngine:
    def __init__(self, inference_engine, db_manager):
        self.targets = {}
        self.track_counter = 0
        self.db_manager = db_manager
        self.profiles = self.db_manager.read_profiles()
        self.mutex = threading.Lock()
        self.inference_engine = inference_engine

    def sync_profiles(self):
        with self.mutex:
            self.profiles = self.db_manager.read_profiles()
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
        current_time = time.time()
        h, w = frame_shape[:2]
        active_ids = set()
        
        with self.mutex:
            for det in detections:
                embedding = det["embedding"]
                box = det["box"]
                shoulders = det["shoulders"]
                det_centroid = ((box[0]+box[2])/2, (box[1]+box[3])/2)
                shoulder_width = np.linalg.norm(np.array(shoulders[0]) - np.array(shoulders[1]))
                
                matched_target = None
                best_cost = float('inf')
                
                for tid, target in self.targets.items():
                    tc = target.get_centroid()
                    centroid_dist = np.linalg.norm(np.array(det_centroid) - np.array(tc))
                    scale_aware_dist = centroid_dist / (shoulder_width + 1e-5)
                    emb_dist = np.linalg.norm(embedding - target.embedding)
                    
                    cost = (scale_aware_dist * 0.5) + (emb_dist * 0.5)
                    if cost <= 0.35 and cost < best_cost:
                        best_cost = cost
                        matched_target = target
                        
                if matched_target is None:
                    track_hash = f"Person_hash_{int(current_time * 1000)}_{self.track_counter}"
                    self.track_counter += 1
                    matched_target = Target(track_hash, embedding, box)
                    
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
                        
                    self.targets[track_hash] = matched_target
                
                dt = current_time - matched_target.last_update
                matched_target.last_seen = current_time
                matched_target.last_update = current_time
                matched_target.box = box
                active_ids.add(matched_target.track_id)
                
                matched_target.pitch = det["pitch"]
                matched_target.yaw = det["yaw"]
                matched_target.roll = det["roll"]
                
                center_shoulder_y = (shoulders[0][1] + shoulders[1][1]) / 2.0
                normalized_shoulder_y = center_shoulder_y / h
                
                if matched_target.name == "Unknown":
                    matched_target.state = "Unregistered Guest - Monitoring Suspended"
                    continue
                    
                gaze_x, is_looking_away = self.inference_engine.extract_pupil_gaze(frame, det["l_eye"], det["r_eye"])
                matched_target.gaze_x = gaze_x
                matched_target.is_looking_away = is_looking_away

                # Calibration Phase
                if matched_target.state == "Calibrating":
                    elapsed_calib = current_time - matched_target.calibration_start
                    matched_target.calibration_pitch_samples.append(matched_target.pitch)
                    matched_target.calibration_shoulder_samples.append(normalized_shoulder_y)
                    
                    if elapsed_calib >= 3.0:
                        matched_target.baseline_pitch = float(np.mean(matched_target.calibration_pitch_samples))
                        matched_target.baseline_shoulder_y = float(np.mean(matched_target.calibration_shoulder_samples))
                        matched_target.state = "Tracking Active"
                    continue
                
                # Active Tracking Phase
                pitch_diff = abs(matched_target.pitch - matched_target.baseline_pitch)
                shoulder_diff = matched_target.baseline_shoulder_y - normalized_shoulder_y
                
                if shoulder_diff > 0.15:
                    matched_target.is_standing = True
                else:
                    matched_target.is_standing = False

                if matched_target.is_standing:
                    matched_target.state = "Standing Mode"
                    matched_target.standing_duration_clock += dt
                    
                    if matched_target.standing_duration_clock >= matched_target.stand_requirement:
                        matched_target.sitting_duration_clock = 0.0
                else:
                    matched_target.sitting_duration_clock += dt
                    if matched_target.sitting_duration_clock >= matched_target.session_limit:
                        matched_target.state = "Session Limit Reached - Stand Up!"
                        matched_target.standing_duration_clock = 0.0
                    else:
                        if matched_target.is_looking_away:
                            matched_target.gaze_away_clock += dt
                            if matched_target.gaze_away_clock > matched_target.gaze_away_limit:
                                matched_target.state = "Ocular Break Recommended"
                            else:
                                matched_target.state = "Looking Away"
                        else:
                            matched_target.gaze_away_clock = 0.0
                            if pitch_diff > matched_target.slouch_sensitivity:
                                matched_target.state = "Posture Deficit Alert"
                                matched_target.slouch_timer += dt
                            else:
                                matched_target.state = "Tracking Active"
                                matched_target.slouch_timer = 0.0
                                    
            to_delete = []
            for tid, target in self.targets.items():
                elapsed = current_time - target.last_seen
                if elapsed > 10.0:
                    to_delete.append(tid)
                    if target.name != "Unknown":
                        self.db_manager.log_metric(target.name, "session_ended", target.sitting_duration_clock)
                elif tid not in active_ids:
                    if target.state != "Calibrating":
                        target.state = "Coasting (Paused)"
                    
            for tid in to_delete:
                del self.targets[tid]
