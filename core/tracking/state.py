import threading
import time
import numpy as np
from collections import deque

state_mutex = threading.Lock()

class UserSession:
    """
    Persistent identity layer that survives temporary track losses.
    """
    def __init__(self, identity_name: str):
        self.identity_name = identity_name
        self.posture_baseline = None
        self.calibrated_baseline_neck_pitch = None
        self.baseline_shoulder_y = None
        self.last_20_embeddings = deque(maxlen=20)
        self.last_seen = time.time()
        self.is_posture_calibrated = False

        # Persistent Temporal State
        self.first_seen_time = time.time()
        self.sitting_duration_clock = 0.0
        self.standing_duration_clock = 0.0
        self.slouch_timer = 0.0
        self.screen_gaze_accumulation_timer = 0.0
        self.ocular_break_timer = 0.0
        
        # Accumulators for Health Evaluation
        self.standing_accumulator_time = 0.0
        self.slouch_accumulator_time = 0.0
        self.active_accumulator_time = 0.0
        
        # Status & Flags
        self.health_status = "Healthy"
        self.ocular_break_announced = False
        self.session_limit_announced = False
        self.slouch_announced = False

    def get_smoothed_embedding(self) -> np.ndarray:
        if not self.last_20_embeddings:
            return None
        # Mean pooling over recent embeddings for pose-robust recognition
        return np.mean(self.last_20_embeddings, axis=0)



class Person:
    """
    Represents a tracked human target containing bounding box logic, biometric identity,
    and temporal state logic for posture evaluation.
    """

    def __init__(self, track_id: int, embedding: np.ndarray, box: list) -> None:
        self.track_id = track_id
        if embedding is None:
            self.embedding = np.zeros(128, dtype=np.float32)
        else:
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

        self.last_20_embeddings = deque(maxlen=20)
        self.is_standing = False
        self.is_looking_away = False
        self.health_status = "Healthy" # local frame copy for logging

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
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)
