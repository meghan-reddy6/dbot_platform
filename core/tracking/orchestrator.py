import logging
import time
import threading
import numpy as np
from config.settings import settings
import multiprocessing
from typing import Any, Optional, Tuple
from detection.person_detector import PersonDetector
from .state import Person
from alerts.alert_manager import AlertManager
from posture.posture_analyzer import PostureAnalyzer
from posture.correction_engine import CorrectionEngine
from .geometry import (
    compute_ious,
    compute_normalized_distances,
    filter_overlapping_detections,
)


def bg_biometric_process_worker(job_queue, result_queue, profiles, config=None):
    if config is None:
        config = {
            "model_base_dir": "D:\\Thundersoft\\dbot",
            "admission_threshold_legacy": 0.880,
        }

    import os
    from recognition.face_recognizer import FaceRecognizer

    model_dir = getattr(config, "model_base_dir", "D:\\Thundersoft\\dbot\\models") if not isinstance(config, dict) else config.get("model_base_dir", "D:\\Thundersoft\\dbot\\models")
    if not model_dir.endswith("models"):
        model_dir = os.path.join(model_dir, "models")
    face_recognizer = FaceRecognizer(model_dir)

    while True:
        try:
            job = job_queue.get()
            if job is None:
                continue

            track_id = job["track_id"]
            frame = job["frame"]
            box = job["box"]
            w, h = job["w"], job["h"]

            x1, y1, x2, y2 = (
                int(max(0, box[0])),
                int(max(0, box[1])),
                int(min(w, box[2])),
                int(min(h, box[3])),
            )
            body_crop = frame[y1:y2, x1:x2]

            embedding = face_recognizer.extract_embedding(body_crop)

            if embedding is not None:
                result_queue.put_nowait({"track_id": track_id, "embedding": embedding})
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.info(f"[MULTIPROCESS WORKER ERROR] {e}")


logger = logging.getLogger(__name__)


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

        self.profiles_json_path = os.path.abspath("profiles_cache.json")
        self.profiles = self.db_manager.load_all_profiles()

        if not os.path.exists(self.profiles_json_path):
            with open(self.profiles_json_path, "w", encoding="utf-8") as f:
                json.dump({}, f)
            logger.info(
                "[OS INITIALIZER] Generated a fresh, empty profiles_cache.json template successfully."
            )

        if os.path.exists(self.profiles_json_path):
            with open(self.profiles_json_path, "r") as f:
                saved_cache = json.load(f)
                for name, data_dict in saved_cache.items():
                    if name not in self.profiles:
                        self.profiles[name] = {}

                    # Fix nested dictionary extraction from profiles_cache.json
                    if isinstance(data_dict, dict) and "embeddings" in data_dict:
                        self.profiles[name]["embeddings"] = data_dict["embeddings"]
                    else:
                        self.profiles[name]["embeddings"] = data_dict
                logger.info(
                    f"[SYSTEM BOOT] Cold-boot telemetry successful! Loaded {len(saved_cache)} profiles from {self.profiles_json_path}"
                )

        if len(self.profiles) == 0:
            logger.info(
                "[OS ENGINE] System Locked. No registered database profiles detected. Please invoke the explicit registration route to initialize the system."
            )
            self.system_locked = True
        else:
            self.system_locked = False

        self.person_detector = PersonDetector()
        self.alert_manager = AlertManager()

        # Multiprocessing Biometric Engine
        self.job_queue = multiprocessing.Queue(maxsize=1)
        self.result_queue = multiprocessing.Queue()
        self.bg_process = multiprocessing.Process(
            target=bg_biometric_process_worker,
            args=(self.job_queue, self.result_queue, self.profiles, settings),
            daemon=True,
        )
        self.bg_process.start()

        self.historical_users = {}
        for profile_name, profile_data in self.profiles.items():
            dummy_box = np.array([0, 0, 0, 0])
            first_emb = profile_data.get("embeddings", [None])[0]
            if first_emb is None and "embedding" in profile_data:
                first_emb = profile_data["embedding"]
            cold_person = Person(
                f"Person_hash_cold_{profile_name}", first_emb, dummy_box
            )
            cold_person.name = profile_name
            cold_person.is_posture_calibrated = True
            self.historical_users[profile_name] = cold_person

        self.tracked_persons = {}
        self.track_id_counter = 0
        self.frame_count = 0

        self.primary_user_track_id = None
        self.last_session_owner = None
        self.current_authenticated_user = None
        self.anchor_lost_frame_counter = 0
        self.last_voice_alert = 0.0

    def initialize_registration_session(self, user_name):
        """Public endpoint to unlock the frame buffer strictly for 5 frames to collect a biometric cluster."""
        self.pending_registration_name = user_name
        logger.info(
            f"[REGISTRATION] Explicit registration session initialized for: {user_name}"
        )
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

            with open(self.profiles_json_path, "r") as f:
                data = json.load(f)

            if not data:
                raise ValueError("JSON file is empty")

            for name, vectors in data.items():
                if name in self.profiles:
                    self.profiles[name]["embeddings"] = [
                        np.array(v, dtype=np.float32) for v in vectors
                    ]
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.profiles = {}
            logger.info(
                f"[PERSISTENCE] No valid profile database found. Starting with a clean slate. ({e})"
            )

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
                            data[name] = [
                                (v.tolist() if isinstance(v, np.ndarray) else v)
                                for v in profile["embeddings"][-15:]
                            ]
                        elif "embedding" in profile:
                            emb = profile["embedding"]
                            data[name] = [
                                emb.tolist() if isinstance(emb, np.ndarray) else emb
                            ]

                target_path = getattr(
                    self,
                    "cache_path",
                    getattr(self, "profiles_json_path", "profiles_cache.json"),
                )
                absolute_path = os.path.abspath(target_path)

                os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
                with open(absolute_path, "w") as f:
                    json.dump(data, f)

                logger.debug(
                    f"[PERSISTENCE] Hard disk write SUCCESSFUL! File generated at: {absolute_path}"
                )
            except (OSError, IOError) as e:
                logger.info(f"[PERSISTENCE] Background save failed: {e}")

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
                    person.gaze_away_limit = float(
                        profile_config_map.get("ocular_break_duration", 20.0)
                    )
                    person.screen_gaze_limit = float(
                        profile_config_map.get("screen_gaze_limit", 1200.0)
                    )
                    person.biometric_cutoff = profile_config_map.get(
                        "biometric_cutoff", 0.55
                    )

    def _match_profile(
        self,
        embedding: np.ndarray,
        spatial_penalty: float = 0.0,
        box: list = None,
        w: int = 1920,
    ) -> Tuple[Optional[str], float]:
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
            cluster_matrix_unit = cluster_matrix / (
                np.linalg.norm(cluster_matrix, axis=1, keepdims=True) + 1e-6
            )

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

    def _dispatch_voice(
        self, text: str, category: str = "general", cooldown: float = 10.0
    ) -> None:
        """Dispatches an asynchronous voice alert through the dedicated daemon queue."""
        if hasattr(self, "alert_manager"):
            self.alert_manager.dispatch(text, category, cooldown)

    def _evaluate_single_target_health(
        self,
        person: "Person",
        pose: dict,
        current_ratio: float,
        dt: float,
        current_time: float,
        frame_shape: tuple,
    ) -> None:
        """
        Evaluates frame-level posture thresholds against calibrated baselines.
        Applies strict temporal hysteresis to prevent state chatter.
        """
        if (
            person.name.startswith("Unknown") or self.primary_user_track_id is None
        ) and person.state != "Calibrating":
            person.next_state = "Unregistered Target"
            person.sustained_slouch_debounce_timer = 0.0
            person.tracking_active_debounce_timer = 0.0
            return

        if person.recovery_calibration_start is not None:
            elapsed_recovery = current_time - person.recovery_calibration_start
            shoulder_width = np.abs(pose["right_shoulder"].x - pose["left_shoulder"].x)
            shoulder_center_y = (
                pose["left_shoulder"].y + pose["right_shoulder"].y
            ) / 2.0
            current_ratio_computed = np.abs(shoulder_center_y - pose["nose"].y) / max(
                shoulder_width, 1e-6
            )
            person.recovery_accumulator.append(
                (current_ratio_computed, shoulder_center_y)
            )

            if elapsed_recovery >= 1.0:
                if person.recovery_accumulator:
                    person.posture_baseline = float(
                        np.mean([i[0] for i in person.recovery_accumulator])
                    )
                    person.baseline_shoulder_y = float(
                        np.mean([i[1] for i in person.recovery_accumulator])
                    )
                person.recovery_calibration_start = None
                person.recovery_accumulator = []
            return

        if person.state == "Calibrating":
            if getattr(person, "calibration_start_time", None) is None:
                person.calibration_start_time = current_time
                person.calibration_accumulator = []
                person.calibration_pitch_acc = []
                person.calibration_y_acc = []
                person.is_posture_calibrated = False

            if not getattr(person, "calibration_announced", False):
                person.calibration_announced = True
                self._dispatch_voice(
                    f"{person.name}, please look straight ahead to calibrate your posture baseline."
                )

            elapsed_calib = current_time - person.calibration_start_time
            shoulder_width = np.abs(pose["right_shoulder"].x - pose["left_shoulder"].x)
            shoulder_center_y = (
                pose["left_shoulder"].y + pose["right_shoulder"].y
            ) / 2.0
            current_ratio_computed = np.abs(shoulder_center_y - pose["nose"].y) / max(
                shoulder_width, 1e-6
            )

            if current_time - getattr(person, "last_log_time", 0.0) >= 1.0:
                person.last_log_time = current_time

            if not hasattr(person, "calibration_accumulator"):
                person.calibration_accumulator = []
            if not hasattr(person, "calibration_pitch_acc"):
                person.calibration_pitch_acc = []
            if not hasattr(person, "calibration_y_acc"):
                person.calibration_y_acc = []

            person.calibration_accumulator.append(current_ratio_computed)
            person.calibration_pitch_acc.append(person.pitch)
            person.calibration_y_acc.append(shoulder_center_y)

            if elapsed_calib >= 3.0:
                if len(person.calibration_accumulator) >= 5:
                    person.posture_baseline = float(
                        np.mean(person.calibration_accumulator)
                    )
                    person.calibrated_baseline_neck_pitch = float(
                        np.mean(person.calibration_pitch_acc)
                    )
                    person.baseline_shoulder_y = float(
                        np.mean(person.calibration_y_acc)
                    )
                    person.is_posture_calibrated = True
                else:
                    if not hasattr(person, "posture_baseline"):
                        person.calibrated_baseline_neck_pitch = 0.0
                        person.posture_baseline = 0.50
                        person.baseline_shoulder_y = (
                            float(np.mean(person.calibration_y_acc))
                            if hasattr(person, "calibration_y_acc")
                            and person.calibration_y_acc
                            else 0.50
                        )
                    person.is_posture_calibrated = True
                    logger.info(
                        f"[CALIBRATION] Timeout fallback triggered. Preserving/Forcing track active for {person.name}."
                    )

                person.calibration_accumulator = []
                person.calibration_pitch_acc = []
                person.calibration_y_acc = []
                person.next_state = "Tracking Active"

                if person.name in self.profiles:
                    if "embeddings" not in self.profiles[person.name]:
                        self.profiles[person.name]["embeddings"] = [
                            self.profiles[person.name]["embedding"]
                        ]
                    self.profiles[person.name]["embeddings"].append(person.embedding)
                    self._save_profiles_json()
                    logger.info(
                        f"[BIOMETRICS] Appended new accessory embedding to profile cluster for {person.name}"
                    )

                self.primary_user_track_id = person.track_id
                self._dispatch_voice(
                    f"Calibration successful for {person.name}. Posture monitoring is now active.",
                    "calibration_success",
                    30.0,
                )

            # SECURE THE CALIBRATION HOLD PARAMETERS: Completely block downstream state evaluation
            return

        if not hasattr(person, "health_status"):
            person.health_status = "Healthy"

        if person.state in [
            "Tracking Active",
            "Standing",
            "Looking Away",
            "Searching / Re-acquiring",
        ]:
            if not person.is_looking_away:
                person.screen_gaze_accumulation_timer += dt
                if person.health_status != "Ocular Break Recommended":
                    person.ocular_break_timer = 0.0

            shoulder_center_y = (
                pose["left_shoulder"].y + pose["right_shoulder"].y
            ) / 2.0

            posture_state = PostureAnalyzer.evaluate(person, pose, frame_shape)
            is_slouching = posture_state.is_slouching
            is_standing = posture_state.is_standing

            person.spine_alignment = posture_state.spine_alignment
            person.shoulder_alignment = posture_state.shoulder_alignment
            person.is_standing = is_standing

            is_pinned_to_ceiling = person.box[1] <= frame_shape[0] * 0.05
            nose_missing = pose["nose"].y <= 0.01

            if not hasattr(person, "standing_accumulator_time"):
                person.standing_accumulator_time = 0.0
            if not hasattr(person, "slouch_accumulator_time"):
                person.slouch_accumulator_time = 0.0
            if not hasattr(person, "active_accumulator_time"):
                person.active_accumulator_time = 0.0

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
                pass  # Wait, standing might still affect p.state depending on the rest of the code, but we'll leave it as a pass for health_status. Wait, no, Standing is a p.state!
                person.next_state = "Standing"
            elif person.slouch_accumulator_time >= 4.0:
                person.health_status = "Posture Deficit Alert"
            elif person.active_accumulator_time >= 1.5:
                if person.state == "Standing":
                    person.next_state = "Tracking Active"
                if person.health_status == "Posture Deficit Alert":
                    person.health_status = "Healthy"

            if person.screen_gaze_accumulation_timer >= person.screen_gaze_limit:
                person.health_status = "Ocular Break Recommended"
                if not getattr(person, "ocular_break_announced", False):
                    logger.info(
                        "[TIMER ALERT] Triggering Voice Alert for: Ocular Break Recommended"
                    )
                    self._dispatch_voice(
                        f"{person.name}, attention, eye strain warning. Please look away from the screen."
                    )
                    person.ocular_break_announced = True

                if person.is_looking_away:
                    person.ocular_break_timer += dt
                    if person.ocular_break_timer >= person.gaze_away_limit:
                        person.screen_gaze_accumulation_timer = 0.0
                        person.ocular_break_timer = 0.0
                        person.health_status = "Healthy"
                        person.ocular_break_announced = False

            if person.state == "Tracking Active":
                person.sitting_duration_clock += dt
                if person.sitting_duration_clock >= person.session_limit:
                    person.health_status = "Session Limit Reached - Stand Up!"
                    if not getattr(person, "session_limit_announced", False):
                        logger.info(
                            "[TIMER ALERT] Triggering Voice Alert for: Session Limit Reached"
                        )
                        self._dispatch_voice(
                            f"{person.name}, you have been sitting for too long. Please stand up."
                        )
                        person.session_limit_announced = True

            if person.health_status == "Posture Deficit Alert":
                if not getattr(person, "slouch_announced", False):
                    advice = CorrectionEngine.get_advice(person)
                    logger.info(
                        f"[TIMER ALERT] Triggering Voice Alert for: Posture Deficit Alert - {advice}"
                    )
                    self._dispatch_voice(
                        f"{advice} {person.name}.",
                        category="posture_alert",
                        cooldown=30.0,
                    )
                    person.slouch_announced = True
            else:
                person.slouch_timer = max(
                    0.0, getattr(person, "slouch_timer", 0.0) - dt
                )
                person.slouch_announced = False

            if person.state == "Standing":
                person.standing_duration_clock += dt
                person.slouch_timer = max(0.0, person.slouch_timer - dt)
                if person.standing_duration_clock >= person.stand_requirement:
                    person.sitting_duration_clock = 0.0
                    person.session_limit_announced = False
                if is_standing:
                    person.baseline_shoulder_y = (person.baseline_shoulder_y * 0.95) + (
                        shoulder_center_y * 0.05
                    )

            if person.state == "Tracking Active":
                fast_frames = getattr(person, "fast_relatch_frames", 0)
                alpha = 0.20 if fast_frames > 0 else 0.005
                person.baseline_shoulder_y = (
                    person.baseline_shoulder_y * (1.0 - alpha)
                ) + (shoulder_center_y * alpha)
                person.posture_baseline = (person.posture_baseline * (1.0 - alpha)) + (
                    current_ratio * alpha
                )
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
        import numpy as np
        from config.settings import settings
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

                        # 1. COMPUTE FRAME CENTER-MASS & SPATIAL ANCHOR GATING
                        frame_width = frame_shape[1]
                        frame_cx = frame_width / 2.0
                        track_cx = (p.box[0] + p.box[2]) / 2.0
                        offset = abs(track_cx - frame_cx)

                        is_in_deadband = offset <= (
                            frame_width * settings.workspace_max_offset_ratio
                        )

                        if not is_in_deadband and p.state != "Tracking Active":
                            p.name = "Unknown [Unregistered Guest]"
                            p.next_state = "Unregistered Guest"
                            p.verification_status = "UNKNOWN"
                            continue

                        if getattr(p, "embedding", None) is None or np.sum(p.embedding) == 0:
                            # Do NOT forcefully log out active users just because a single frame is blurry.
                            # Just skip biometric verification for this frame and maintain current state.
                            continue

                        # (Anti-Drift Hooking has been removed to prevent blind session hijacking)

                        # 3. REFACTOR THE EMBEDDING COMPARISON CORE
                        candidate_name = "Unknown"
                        sim = 0.0
                        margin = 1.0

                        if (
                            getattr(p, "embedding", None) is not None
                            and getattr(self, "profiles", None)
                            and len(self.profiles) > 0
                        ):
                            highest_consensus = -1.0
                            second_highest_consensus = -1.0
                            matched_db_profile_string = "Unknown"

                            norm_embedding = p.embedding / (
                                np.linalg.norm(p.embedding) + 1e-6
                            )

                            for name, profile in self.profiles.items():
                                db_embs = profile.get("embeddings", [])
                                if "embedding" in profile and len(db_embs) == 0:
                                    db_embs = [profile["embedding"]]

                                if len(db_embs) == 0:
                                    continue

                                templates = np.array(db_embs, dtype=np.float32)
                                norm_templates = templates / (
                                    np.linalg.norm(templates, axis=1, keepdims=True)
                                    + 1e-6
                                )
                                similarities = np.dot(norm_templates, norm_embedding)

                                max_sim = np.max(similarities)
                                if max_sim > highest_consensus:
                                    second_highest_consensus = highest_consensus
                                    highest_consensus = max_sim
                                    matched_db_profile_string = name
                                elif max_sim > second_highest_consensus:
                                    second_highest_consensus = max_sim

                            sim = highest_consensus
                            margin = (
                                highest_consensus - max(second_highest_consensus, 0.0)
                                if len(self.profiles) > 1
                                else 1.0
                            )

                            if len(self.profiles) > 1 and margin < 0.06:
                                candidate_name = "Unknown [Collision Risk Reject]"
                            elif sim < settings.admission_threshold_strict:
                                candidate_name = "Unknown [Low Confidence]"
                            else:
                                candidate_name = matched_db_profile_string

                        current_area = (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])
                        min_primary_area = (
                            frame_shape[0] * frame_shape[1]
                        ) * settings.workspace_min_area_ratio

                        if (
                            candidate_name
                            not in [
                                "Unknown",
                                "Unknown [Collision Risk Reject]",
                                "Unknown [Low Confidence]",
                                "Unknown [Unregistered Guest]",
                            ]
                            and current_area < min_primary_area
                        ):
                            p.name = "Unknown [Unregistered Guest]"
                            p.state = "Unregistered Guest"
                            p.verification_status = "UNKNOWN"
                            p.biometric_match_counter = 0
                            continue

                        if getattr(
                            p, "verification_status", "UNKNOWN"
                        ) == "VERIFIED" and p.track_id == getattr(
                            self, "primary_user_track_id", None
                        ):
                            sim_threshold = settings.hysteresis_holding_threshold
                        else:
                            sim_threshold = settings.admission_threshold_strict

                        is_valid_candidate = candidate_name not in [
                            "Unknown",
                            "Unknown [Collision Risk Reject]",
                            "Unknown [Low Confidence]",
                            "Unknown [Unregistered Guest]",
                        ]

                        if is_valid_candidate and sim >= sim_threshold:
                            # Absolute Global Name Uniqueness
                            if (
                                self.current_authenticated_user
                                and candidate_name == self.current_authenticated_user
                                and getattr(self, "primary_user_track_id", None) is not None
                                and p.track_id != self.primary_user_track_id
                            ):
                                p.name = "Unknown [Unregistered Guest]"
                                p.state = "Unregistered Guest"
                                p.verification_status = "UNKNOWN"
                                p.biometric_match_counter = 0
                                continue

                            p.candidate_name = candidate_name
                            p.biometric_match_counter = (
                                getattr(p, "biometric_match_counter", 0) + 1
                            )
                            if p.biometric_match_counter >= 5:
                                p.verification_status = "VERIFIED"

                                # STRICT SEAT COORDINATE ISOLATION
                                if (
                                    p.candidate_name == getattr(self, "current_authenticated_user", None)
                                    and getattr(self, "primary_user_track_id", None) is not None
                                    and p.track_id != self.primary_user_track_id
                                ):
                                    p.name = "Unknown [Unregistered Guest]"
                                    p.verification_status = "UNKNOWN"
                                    p.next_state = "Unregistered Guest"
                                else:
                                    p.name = p.candidate_name
                                    p.verified_name = p.candidate_name
                                    p.frame_val_name = p.candidate_name
                                    p.next_state = "Tracking Active"

                        elif "Unknown" in candidate_name:
                            if (
                                getattr(self, "pending_registration_name", None)
                                is not None
                            ):
                                p.candidate_name = "Unknown"

                                # Accumulate embeddings for multi-angle enrollment
                                if not hasattr(p, "embedding_cluster"):
                                    p.embedding_cluster = []
                                p.embedding_cluster.append(res["embedding"])

                                p.biometric_match_counter = (
                                    getattr(p, "biometric_match_counter", 0) + 1
                                )
                                if p.biometric_match_counter >= 5:
                                    new_name = self.pending_registration_name

                                    if hasattr(self, "db_manager"):
                                        try:
                                            import numpy as np

                                            composite_embedding = np.mean(
                                                np.array(p.embedding_cluster), axis=0
                                            ).astype(np.float32)

                                            p.embedding = composite_embedding

                                            self.db_manager.create_profile(
                                                new_name, composite_embedding
                                            )

                                            serializable_cluster = [
                                                emb.tolist()
                                                if hasattr(emb, "tolist")
                                                else list(emb)
                                                for emb in p.embedding_cluster
                                            ]

                                            self.profiles[new_name] = {
                                                "embeddings": serializable_cluster
                                            }

                                            with open(
                                                self.profiles_json_path,
                                                "w",
                                                encoding="utf-8",
                                            ) as f:
                                                import json

                                                json.dump(self.profiles, f, indent=4)

                                            if (
                                                hasattr(self, "bg_process")
                                                and self.bg_process.is_alive()
                                            ):
                                                self.bg_process.terminate()
                                                self.bg_process.join(timeout=1.0)
                                            import multiprocessing

                                            self.bg_process = multiprocessing.Process(
                                                target=bg_biometric_process_worker,
                                                args=(
                                                    self.job_queue,
                                                    self.result_queue,
                                                    self.profiles,
                                                    settings,
                                                ),
                                                daemon=True,
                                            )
                                            self.bg_process.start()

                                            self.system_locked = False
                                            self.pending_registration_name = None
                                        except Exception:
                                            self.pending_registration_name = None

                                        p.verification_status = "VERIFIED"
                                        p.name = new_name
                                        p.verified_name = new_name
                                        p.frame_val_name = new_name
                                        p.next_state = "Tracking Active"
                                    else:
                                        p.next_state = "Candidate"
                            else:
                                p.name = "Unknown [Unregistered Guest]"
                                p.verification_status = "UNKNOWN"
                        else:
                            # ELIMINATE AUTHENTICATED FALLBACKS or Collisions
                            if p.track_id != getattr(
                                self, "primary_user_track_id", None
                            ):
                                p.name = "Unknown [Unregistered Guest]"
                                p.verification_status = "UNKNOWN"
                                p.next_state = "Unregistered Guest"
                            else:
                                # Instant safety reset on a bad or suspicious frame
                                p.biometric_match_counter = 0
                                p.next_state = "Searching / Re-acquiring"
            except python_queue.Empty:
                break

        # Handle manual recalibration events cleanly...
        with self.mutex:
            if getattr(self, "trigger_recalibration", False) or getattr(
                self, "manual_recalibration_requested", False
            ):
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
                self.trigger_recalibration = False
                self.manual_recalibration_requested = False

        # ---------------------------------------------------------
        # 2. DETECTION & SPATIAL ASSIGNMENT (FAST PATH)
        # ---------------------------------------------------------
        detections = self.person_detector.detect(frame)
        active_track_ids = set()

        if detections is not None and len(detections) > 0:
            verified_tracks = [
                p
                for p in self.tracked_persons.values()
                if p.verification_status == "VERIFIED"
            ]
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
                        best_cost = float("inf")
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
                        track_hash = (
                            mot_id
                            if mot_id is not None
                            else f"Guest_hash_{int(current_time * 1000)}_{self.track_id_counter}"
                        )
                        self.track_id_counter += 1
                        best_guest = Person(
                            track_hash, np.zeros(128, dtype=np.float32), box
                        )
                        best_guest.name = "Unknown"
                        best_guest.state = "Unregistered Guest"
                        best_guest.next_state = "Unregistered Guest"
                        best_guest.verification_status = "UNKNOWN"
                        self.tracked_persons[track_hash] = best_guest
                        det["matched_person"] = best_guest
                        active_track_ids.add(best_guest.track_id)

                    # Update Person with Detection
                    best_guest.last_seen = current_time
                    dt = (
                        current_time - best_guest.last_update
                        if best_guest.last_update
                        else 0.033
                    )
                    best_guest.last_update = current_time
                    best_guest.box = box
                    best_guest.pose = det.get("pose", getattr(best_guest, "pose", None))
                    best_guest.pitch = (best_guest.pitch * 0.6) + (
                        det.get("pitch", best_guest.pitch) * 0.4
                    )
                    best_guest.yaw = det.get("yaw", best_guest.yaw)
                    best_guest.roll = det.get("roll", best_guest.roll)

                    pose = best_guest.pose
                    if pose:
                        shoulder_width = np.abs(
                            pose["left_shoulder"].x - pose["right_shoulder"].x
                        )
                        shoulder_center_y = (
                            pose["left_shoulder"].y + pose["right_shoulder"].y
                        ) / 2.0
                        current_ratio = np.abs(
                            shoulder_center_y - pose["nose"].y
                        ) / max(shoulder_width, 1e-6)
                        best_guest.last_y = shoulder_center_y

                        alpha = 0.10
                        if best_guest.smoothed_pitch is None:
                            best_guest.smoothed_pitch = best_guest.pitch
                            best_guest.smoothed_ratio = current_ratio
                            best_guest.smoothed_y = shoulder_center_y
                        else:
                            best_guest.smoothed_pitch = (
                                1 - alpha
                            ) * best_guest.smoothed_pitch + alpha * best_guest.pitch
                            best_guest.smoothed_ratio = (
                                1 - alpha
                            ) * best_guest.smoothed_ratio + alpha * current_ratio
                            best_guest.smoothed_y = (
                                1 - alpha
                            ) * best_guest.smoothed_y + alpha * shoulder_center_y

                        if (
                            best_guest.track_id == self.primary_user_track_id
                            and "l_eye" in det
                            and "r_eye" in det
                        ):
                            gaze_x, is_looking_away = (
                                self.person_detector.extract_pupil_gaze(
                                    frame, det["l_eye"], det["r_eye"]
                                )
                            )
                            best_guest.gaze_x = gaze_x
                            best_guest.is_looking_away = is_looking_away
                            self._evaluate_single_target_health(
                                best_guest,
                                pose,
                                current_ratio,
                                dt,
                                current_time,
                                frame_shape,
                            )

                    # Background Cluster Accumulation
                    if (
                        best_guest.verification_status == "VERIFIED"
                        and best_guest.state == "Tracking Active"
                    ):
                        profile = self.profiles.get(best_guest.name)
                        if (
                            profile
                            and "embeddings" in profile
                            and len(profile["embeddings"]) < 15
                            and hasattr(best_guest, "embedding")
                            and np.sum(best_guest.embedding) != 0
                        ):
                            db_embs = profile["embeddings"]
                            norm_live = best_guest.embedding / (
                                np.linalg.norm(best_guest.embedding) + 1e-6
                            )
                            max_sim = 0.0
                            for db_emb in db_embs:
                                db_emb_np = np.array(db_emb, dtype=np.float32)
                                norm_db = db_emb_np / (np.linalg.norm(db_emb_np) + 1e-6)
                                sim = np.dot(norm_live, norm_db)
                                if sim > max_sim:
                                    max_sim = sim
                            if 0.78 <= max_sim <= 0.93:
                                profile["embeddings"].append(
                                    best_guest.embedding.tolist()
                                )
                                self._save_profiles_json()

                    # Analytics Flush
                    if (
                        current_time
                        - getattr(best_guest, "last_analytics_flush_time", 0.0)
                        >= 10.0
                    ):
                        self.db_manager.log_analytics_flush(
                            user_name=best_guest.name,
                            session_state=best_guest.state,
                            duration_seconds=10.0,
                            continuous_sitting_seconds=best_guest.sitting_duration_clock,
                            continuous_gaze_seconds=best_guest.screen_gaze_accumulation_timer,
                            average_head_pitch=getattr(
                                best_guest, "smoothed_pitch", best_guest.pitch
                            ),
                            ocular_break_accumulator=best_guest.ocular_break_timer,
                            spine_alignment=getattr(best_guest, "spine_alignment", 0.0),
                            shoulder_alignment=getattr(
                                best_guest, "shoulder_alignment", 0.0
                            ),
                        )
                        best_guest.last_analytics_flush_time = current_time

        # ---------------------------------------------------------
        # 2b. PRIMARY USER PROXIMITY GUARD (ANTI-HIJACKING)
        # ---------------------------------------------------------
        active_tracks = [
            p for p in self.tracked_persons.values() if p.track_id in active_track_ids
        ]
        verified_tracks = [
            p
            for p in active_tracks
            if p.verification_status == "VERIFIED" and p.state == "Tracking Active"
        ]
        unverified_tracks = [
            p
            for p in active_tracks
            if getattr(p, "verification_status", "UNKNOWN") != "VERIFIED"
        ]

        for v_track in verified_tracks:
            v_area = (v_track.box[2] - v_track.box[0]) * (
                v_track.box[3] - v_track.box[1]
            )
            for u_track in unverified_tracks:
                u_area = (u_track.box[2] - u_track.box[0]) * (
                    u_track.box[3] - u_track.box[1]
                )
                if u_area > (v_area * 1.5):
                    v_track.state = "Unregistered Guest"
                    v_track.next_state = "Unregistered Guest"
                    v_track.name = "Unknown"
                    v_track.verification_status = "UNKNOWN"
                    v_track.biometric_match_counter = 0

                    u_track.biometric_match_counter = 0
                    u_track.state = "Verifying Identity"
                    u_track.next_state = "Verifying Identity"

                    if getattr(self, "primary_user_track_id", None) == v_track.track_id:
                        self.primary_user_track_id = None
                        self.current_authenticated_user = None
                    break

        # ---------------------------------------------------------
        # 3. BIOMETRIC OFFLOAD (ASYNC SUBMISSION)
        # ---------------------------------------------------------
        is_profile_already_active = any(
            getattr(p, "verification_status", "UNKNOWN") == "VERIFIED"
            and p.state == "Tracking Active"
            for p in self.tracked_persons.values()
        )

        if self.job_queue.empty() and not is_profile_already_active:
            for act_id in active_track_ids:
                person = self.tracked_persons[act_id]
                if getattr(person, "verification_status", "UNKNOWN") != "VERIFIED":
                    try:
                        self.job_queue.put_nowait(
                            {
                                "track_id": person.track_id,
                                "frame": frame.copy(),
                                "box": person.box,
                                "w": frame_shape[1],
                                "h": frame_shape[0],
                            }
                        )
                        person.next_state = "Verifying Identity"
                        person.verification_status = "VERIFYING"
                        break  # Only queue one job at a time
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
                    if person.name and person.name not in [
                        "Unknown / Bystander",
                        "Unknown (Ready for Registration)",
                        "Unknown",
                    ]:
                        self.db_manager.log_session_metrics(
                            person.name, "session_ended", person.sitting_duration_clock
                        )
                    keys_to_delete.append(track_id)
                elif time_since_seen > 2.0:
                    if person.state not in ["Calibrating", "Absent"]:
                        person.next_state = "Searching / Re-acquiring"
            else:
                if person.verification_status == "VERIFIED":
                    if person.state in ["Absent", "Searching / Re-acquiring"]:
                        person.next_state = "Tracking Active"
                        person.lost_grace_timer = 0.0
                elif person.verification_status == "UNKNOWN":
                    person.next_state = "Unregistered Guest"

            # State Emitting & Dampening
            target = getattr(person, "next_state", person.state)

            if target != person.state:
                if getattr(person, "target_next_state", None) == target:
                    person.frames_in_current_state = (
                        getattr(person, "frames_in_current_state", 0) + 1
                    )
                else:
                    person.target_next_state = target
                    person.frames_in_current_state = 1

                # Determine transition threshold
                threshold = 0
                if target == "Verifying Identity":
                    threshold = 5
                elif target == "Unregistered Guest":
                    threshold = 30

                # Commit state transition if sustained threshold met
                if person.frames_in_current_state >= threshold:
                    person.state = target
                    person.frames_in_current_state = 0
            else:
                person.target_next_state = None
                person.frames_in_current_state = 0

            if person.state != getattr(person, "last_logged_state", None):
                logger.info(f"[STATE CHANGE] Track {person.track_id} -> {person.state}")
                person.last_logged_state = person.state
                person.last_state = person.state

        for k in keys_to_delete:
            del self.tracked_persons[k]

        # ---------------------------------------------------------
        # 5. DYNAMIC AUTHENTICATED USER SELECTION
        # ---------------------------------------------------------
        frame_width = frame_shape[1]
        frame_cx = frame_width / 2.0

        closest_user = None
        closest_dist = float("inf")

        for p in self.tracked_persons.values():
            if p.state == "Tracking Active" and p.verification_status == "VERIFIED":
                track_cx = (p.box[0] + p.box[2]) / 2.0
                dist = abs(track_cx - frame_cx)

                if dist <= (frame_width * settings.workspace_max_offset_ratio):
                    if dist < closest_dist:
                        closest_dist = dist
                        closest_user = p

        if closest_user:
            self.anchor_lost_frame_counter = 0
            current_occupant = closest_user.name
        else:
            self.anchor_lost_frame_counter += 1
            if self.anchor_lost_frame_counter > 60:
                current_occupant = None
            else:
                current_occupant = getattr(self, "current_authenticated_user", None)

        if current_occupant != getattr(self, "last_session_owner", None):
            if current_occupant is not None:
                logger.info(f"[SESSION] Primary seat occupied by: {current_occupant}")
            else:
                logger.info("[SESSION] Primary seat vacated.")
            self.last_session_owner = current_occupant

        if closest_user:
            self.current_authenticated_user = closest_user.name
            self.primary_user_track_id = closest_user.track_id
        else:
            if self.anchor_lost_frame_counter > 60:
                self.current_authenticated_user = None
            self.primary_user_track_id = None
