import logging
import os
import sqlite3
import time
from typing import Any

import numpy as np

from alerts.alert_manager import AlertManager
from config.paths import repo_path, uses_default_data_dir
from config.settings_manager import settings
from database.crud import ProfileExistsError
from detection.person_detector import PersonDetector

from .biometric_worker import BiometricWorker
from .geometry import (
    compute_ious,
    compute_normalized_distances,
    filter_overlapping_detections,
    head_crop_box,
)
from .health import HealthMonitorMixin
from .identity import classify_embedding, matching_profile
from .state import Person, UserSession, state_mutex
from .states import (
    ACTIVE_STATES,
    UNKNOWN_CANDIDATES,
    UNKNOWN_NAME,
    UNREGISTERED_GUEST_NAME,
    HealthStatus,
    TrackState,
    Verification,
)

logger = logging.getLogger(__name__)

# How long a new target state must persist before it is committed (formerly 5 / 30 frames at ~30 FPS).
_STATE_HOLD_SECONDS = {TrackState.VERIFYING: 0.15, TrackState.UNREGISTERED: 1.0}
_ANCHOR_LOST_SECONDS = 2.0
_SEARCH_AFTER_SECONDS = 2.0
_EVICT_AFTER_SECONDS = 5.0
_SESSION_TTL_SECONDS = 300.0
_ANALYTICS_INTERVAL_SECONDS = 10.0
_REGISTRATION_TIMEOUT_SECONDS = 60.0
_REGISTRATION_SAMPLES = 5
_NO_SESSION_METRICS = {"Unknown / Bystander", "Unknown (Ready for Registration)", UNKNOWN_NAME}


class TrackerEngine(HealthMonitorMixin):
    """
    Tracks people across frames, binds tracks to registered identities through the face worker, and
    runs the health monitor for the primary user. Every method expects `self.mutex` to be held.
    """

    def __init__(self, db_manager: Any, person_detector=None, alert_manager=None) -> None:
        self.mutex = state_mutex
        self.db_manager = db_manager
        self.last_system_message = "System Boot Sequence Complete. Awaiting targets..."

        if uses_default_data_dir():
            self._import_legacy_embedding_cache()
        self.profiles = self.db_manager.load_all_profiles()
        logger.info(f"[SYSTEM BOOT] Loaded {len(self.profiles)} registered profiles.")

        self.person_detector = person_detector or PersonDetector()
        self.alert_manager = alert_manager or AlertManager()
        self.worker = BiometricWorker()

        self.tracked_persons = {}
        self.track_id_counter = 0

        self.primary_user_track_id = None
        self.last_session_owner = None
        self.current_authenticated_user = None
        self.anchor_lost_since = None

        # Persistent Identity Layer
        self.active_sessions = {}

        self.pending_registration_name = None
        self.pending_registration_deadline = 0.0
        self.manual_recalibration_requested = False
        self.manual_session_restart_requested = False

    def _import_legacy_embedding_cache(self):
        # Older builds wrote embeddings to profiles_cache.json in the repository root.
        path = repo_path("profiles_cache.json")
        if os.path.exists(path):
            self.db_manager.import_legacy_embedding_cache(path)

    # ------------------------------------------------------------------
    # API used by the HTTP layer (each call takes the state lock)
    # ------------------------------------------------------------------

    def worker_status(self) -> dict:
        return self.worker.status()

    def shutdown(self):
        self.worker.shutdown()

    def request_registration(self, name):
        """Queues enrolment of the unregistered person in view; returns why that is impossible, or None."""
        with self.mutex:
            now = time.monotonic()
            in_view = [p for p in self.tracked_persons.values() if now - p.last_seen < 1.0]
            if any(p.verification_status == Verification.VERIFIED and p.state in ACTIVE_STATES for p in in_view):
                return "A registered user is in view. Only the person to register should face the camera."
            if sum(p.verification_status != Verification.VERIFIED for p in in_view) > 1:
                return "More than one unregistered person is in view. Only the person to register should face the camera."
            self.pending_registration_name = name
            self.pending_registration_deadline = now + _REGISTRATION_TIMEOUT_SECONDS
            return None

    def request_recalibration(self):
        with self.mutex:
            self.manual_recalibration_requested = True

    def request_session_restart(self):
        with self.mutex:
            self.manual_session_restart_requested = True

    def remove_profile(self, name) -> bool:
        """Deletes the profile with its telemetry and forgets the identity everywhere in memory."""
        with self.mutex:
            removed = self.db_manager.delete_profile(name)
            self.profiles.pop(name, None)
            self.active_sessions.pop(name, None)
            if self.pending_registration_name == name:
                self.pending_registration_name = None
            for person in self.tracked_persons.values():
                if person.name == name:
                    person.name = UNKNOWN_NAME
                    person.verification_status = Verification.UNKNOWN
                    person.state = TrackState.UNREGISTERED
                    person.next_state = TrackState.UNREGISTERED
                    person.biometric_match_counter = 0
                    person.state_history_window.clear()
                    if self.primary_user_track_id == person.track_id:
                        self.primary_user_track_id = None
            if self.current_authenticated_user == name:
                self.current_authenticated_user = None
            return removed

    def sync_profiles(self) -> None:
        """Reloads profiles from the database and re-applies per-user limits to live tracks."""
        with self.mutex:
            self.profiles = self.db_manager.load_all_profiles()
            for person in self.tracked_persons.values():
                self._apply_profile_limits(person)

    def _apply_profile_limits(self, person) -> None:
        profile = self.profiles.get(person.name)
        if not profile:
            return
        sensitivity = profile.get("slouch_sensitivity")
        if sensitivity is None:
            sensitivity = settings.get("neck_pitch_tolerance_degrees", 15.0)
        person.slouch_sensitivity = float(sensitivity)
        person.session_limit = profile.get("session_limit", 1200)
        person.stand_requirement = profile.get("stand_requirement", 180)
        person.gaze_away_limit = float(profile.get("ocular_break_duration", 20.0))
        person.screen_gaze_limit = float(profile.get("screen_gaze_limit", 1200.0))

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    def process_frame_mot(self, frame, frame_shape):
        """
        Executes multi-object tracking and health evaluations on each incoming frame.
        """
        # Inference touches no shared state, so it runs before the lock the API also uses.
        detections = self.person_detector.detect(frame)
        with self.mutex:
            self._process_frame_locked(frame, frame_shape, detections)

    def _process_frame_locked(self, frame, frame_shape, detections):
        current_time = time.monotonic()
        if self.worker.supervise(current_time):
            # The restarted worker never saw the jobs these tracks are waiting on.
            for person in self.tracked_persons.values():
                if person.verification_status == Verification.VERIFYING:
                    person.verification_status = Verification.UNKNOWN
        self._expire_pending_registration(current_time)

        self._ingest_biometric_results(frame_shape, current_time)
        self._apply_manual_requests(current_time)
        active_track_ids = self._assign_detections(frame, frame_shape, detections, current_time)
        self._guard_primary_seat(active_track_ids)
        self._submit_biometric_job(frame, frame_shape, active_track_ids)
        self._update_lifecycle(active_track_ids, current_time)
        self._select_primary_user(frame_shape, current_time)

    # ------------------------------------------------------------------
    # 1. Identity: results from the face worker
    # ------------------------------------------------------------------

    def _expire_pending_registration(self, now):
        if self.pending_registration_name is None or now < self.pending_registration_deadline:
            return
        logger.warning(f"[REGISTRATION] Timed out waiting for a face to enrol as {self.pending_registration_name}.")
        self.last_system_message = f"Registration of {self.pending_registration_name} timed out; no face was captured."
        self.pending_registration_name = None
        for person in self.tracked_persons.values():
            person.embedding_cluster = []

    def _ingest_biometric_results(self, frame_shape, current_time):
        for result in self.worker.results():
            person = self.tracked_persons.get(result["track_id"])
            if person is not None:
                self._handle_biometric_result(person, result, frame_shape, current_time)

    def _handle_biometric_result(self, p, res, frame_shape, current_time):
        if "embedding" in res:
            p.embedding = res["embedding"]
            p.last_20_embeddings.append(res["embedding"])

        # Only someone in the central workspace can be identified (or enrolled).
        frame_width = frame_shape[1]
        offset = abs((p.box[0] + p.box[2]) / 2.0 - frame_width / 2.0)
        if offset > frame_width * settings.workspace_max_offset_ratio and p.state != TrackState.TRACKING:
            p.name = UNREGISTERED_GUEST_NAME
            p.next_state = TrackState.UNREGISTERED
            p.verification_status = Verification.UNKNOWN
            return

        if getattr(p, "embedding", None) is None or np.sum(p.embedding) == 0:
            # A blurry frame must not log out an active user; keep the current state.
            return

        candidate_name = UNKNOWN_NAME
        if self.profiles:
            smoothed = np.mean(p.last_20_embeddings, axis=0) if p.last_20_embeddings else p.embedding
            candidate_name, best_name, sim, margin = classify_embedding(
                smoothed, self.profiles, settings.recognition_threshold_initial, settings.ambiguity_margin
            )
            box_w, box_h = p.box[2] - p.box[0], p.box[3] - p.box[1]
            logger.debug(
                f"[BIOMETRICS] Track={p.track_id} | Box={box_w:.1f}x{box_h:.1f} | Top={best_name} ({sim:.3f}) "
                f"| Margin={margin:.3f} | Result={candidate_name}"
            )

        is_valid_candidate = candidate_name not in UNKNOWN_CANDIDATES
        area = (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])
        if is_valid_candidate and area < frame_shape[0] * frame_shape[1] * settings.workspace_min_area_ratio:
            # Too small to be the person sitting at this desk.
            p.name = UNREGISTERED_GUEST_NAME
            p.state = TrackState.UNREGISTERED
            p.verification_status = Verification.UNKNOWN
            p.biometric_match_counter = 0
            return

        if is_valid_candidate:
            self._confirm_identity(p, candidate_name, current_time)
        elif self.pending_registration_name is not None:
            self._collect_registration_sample(p, res)
        else:
            p.name = UNREGISTERED_GUEST_NAME
            p.verification_status = Verification.UNKNOWN

    def _confirm_identity(self, p, candidate_name, current_time):
        p.candidate_name = candidate_name
        p.biometric_match_counter = getattr(p, "biometric_match_counter", 0) + 1
        if p.biometric_match_counter < getattr(settings, "confirmation_frame_count", 5):
            return
        p.verification_status = Verification.VERIFIED

        if candidate_name not in self.active_sessions:
            logger.info(f"[TRACK DEBUG] Creating new UserSession for {candidate_name}")
            self.active_sessions[candidate_name] = UserSession(candidate_name)
        else:
            logger.info(f"[TRACK DEBUG] Reusing existing UserSession for {candidate_name}")
        session = self.active_sessions[candidate_name]
        session.last_seen = current_time

        if any(
            t.name == candidate_name and t.track_id != p.track_id and t.state not in (TrackState.SEARCHING, TrackState.ABSENT)
            for t in self.tracked_persons.values()
        ):
            # Another live track already is this user: one identity may hold only one seat.
            p.name = UNREGISTERED_GUEST_NAME
            p.verification_status = Verification.UNKNOWN
            p.next_state = TrackState.UNREGISTERED
            return

        old_primary_id = self.primary_user_track_id
        if old_primary_id is not None and old_primary_id != p.track_id:
            old_track = self.tracked_persons.get(old_primary_id)
            if old_track and old_track.name == candidate_name:
                logger.info(
                    f"\n[SESSION MIGRATION]\n"
                    f"Identity: {candidate_name}\n"
                    f"Action: Transferring primary seat to new track ID {p.track_id}\n"
                    f"Monitoring: CONTINUED\n"
                )
                old_track.last_seen = 0  # Force garbage collection
                self.primary_user_track_id = p.track_id
            elif not old_track and session.identity_name == candidate_name:
                # The primary track was evicted, but this person matched the primary session.
                self.primary_user_track_id = p.track_id
        elif old_primary_id is None:
            self.primary_user_track_id = p.track_id

        p.name = candidate_name
        p.verified_name = candidate_name
        p.frame_val_name = candidate_name
        self._apply_profile_limits(p)

        if session.is_posture_calibrated:
            logger.info("[CALIBRATION DECISION] IDENTITY_ALREADY_CALIBRATED. Bypassing recalibration.")
            p.posture_baseline = session.posture_baseline
            p.calibrated_baseline_neck_pitch = session.calibrated_baseline_neck_pitch
            p.baseline_shoulder_y = session.baseline_shoulder_y
            p.is_posture_calibrated = True
            p.next_state = TrackState.TRACKING
        else:
            logger.info("[CALIBRATION DECISION] NEW_PERSON or session uncalibrated.")
            p.next_state = TrackState.CALIBRATING

    def _collect_registration_sample(self, p, res):
        p.candidate_name = UNKNOWN_NAME
        # Several samples give the profile more than one view of the face.
        if not hasattr(p, "embedding_cluster"):
            p.embedding_cluster = []
        p.embedding_cluster.append(res["embedding"])
        p.biometric_match_counter = getattr(p, "biometric_match_counter", 0) + 1
        if p.biometric_match_counter < _REGISTRATION_SAMPLES:
            return

        new_name = self.pending_registration_name
        cluster = np.array(p.embedding_cluster, dtype=np.float32)
        composite_embedding = cluster.mean(axis=0).astype(np.float32)
        p.embedding_cluster = []
        p.biometric_match_counter = 0
        self.pending_registration_name = None

        existing = matching_profile(composite_embedding, self.profiles, settings.recognition_threshold_initial)
        if existing is not None:
            logger.warning(f"[REGISTRATION] Refused {new_name}: face matches existing profile {existing}.")
            self.last_system_message = f"Not registered: this face already matches the profile {existing}."
            return
        try:
            self.db_manager.create_profile(new_name, composite_embedding, embeddings=cluster)
        except (ProfileExistsError, sqlite3.Error) as e:
            logger.error(f"[REGISTRATION] Could not register {new_name}: {e}")
            return

        self.profiles = self.db_manager.load_all_profiles()
        p.embedding = composite_embedding
        p.verification_status = Verification.VERIFIED
        p.name = new_name
        p.verified_name = new_name
        p.frame_val_name = new_name
        self._apply_profile_limits(p)
        p.next_state = TrackState.CALIBRATING
        logger.info(f"[REGISTRATION] Registered {new_name}.")

    # ------------------------------------------------------------------
    # 2. Requests queued by the dashboard
    # ------------------------------------------------------------------

    def _apply_manual_requests(self, current_time):
        if self.manual_recalibration_requested:
            self.manual_recalibration_requested = False
            person = self.tracked_persons.get(self.primary_user_track_id)
            if person is not None:
                logger.info(
                    f"\n[MANUAL RECALIBRATION]\n"
                    f"Identity: {person.name}\n"
                    f"Action: RESETTING BASELINE\n"
                    f"Result: CALIBRATION STARTED\n"
                )
                person.calibration_start_time = current_time
                person.calibration_accumulator = []
                person.calibration_pitch_acc = []
                person.calibration_y_acc = []
                person.next_state = TrackState.CALIBRATING
                person.state = TrackState.CALIBRATING
                person.is_posture_calibrated = False
                person.calibration_announced = False

                session = self.active_sessions.get(person.name)
                if session:
                    session.is_posture_calibrated = False
                    session.posture_baseline = None
                    session.calibrated_baseline_neck_pitch = None
                    session.baseline_shoulder_y = None

        if self.manual_session_restart_requested:
            self.manual_session_restart_requested = False
            person = self.tracked_persons.get(self.primary_user_track_id)
            session = self.active_sessions.get(person.name) if person is not None else None
            if session:
                logger.info(f"[SESSION RESTART] Restarting session timers for {person.name}")
                session.sitting_duration_clock = 0.0
                session.standing_duration_clock = 0.0
                session.slouch_timer = 0.0
                session.screen_gaze_accumulation_timer = 0.0
                session.ocular_break_timer = 0.0
                session.session_limit_announced = False
                session.health_status = HealthStatus.HEALTHY
                session.first_seen_time = current_time

    # ------------------------------------------------------------------
    # 3. Detections -> tracks
    # ------------------------------------------------------------------

    def _assign_detections(self, frame, frame_shape, detections, current_time):
        """Matches detections to tracks (creating new ones), updates them, and returns the IDs seen."""
        active_track_ids = set()
        if detections is None or len(detections) == 0:
            return active_track_ids

        verified_tracks = [p for p in self.tracked_persons.values() if p.verification_status == Verification.VERIFIED]
        detections = filter_overlapping_detections(detections, verified_tracks)
        if len(detections) == 0:
            return active_track_ids

        remaining_tracks = list(self.tracked_persons.values())
        if remaining_tracks:
            det_boxes = np.array([d["box"] for d in detections])
            trk_boxes = np.array([p.box for p in remaining_tracks])
            ious = compute_ious(det_boxes, trk_boxes)
            norm_dists = compute_normalized_distances(det_boxes, trk_boxes)

        for i, det in enumerate(detections):
            mot_id = det.get("track_id")
            person = None
            if mot_id is not None and mot_id in self.tracked_persons:
                person = self.tracked_persons[mot_id]
            else:
                best_cost = float("inf")
                for j, candidate in enumerate(remaining_tracks):
                    if candidate.track_id in active_track_ids:
                        continue
                    if (ious[i, j] > 0.45 or norm_dists[i, j] < 0.25) and norm_dists[i, j] < best_cost:
                        best_cost = norm_dists[i, j]
                        person = candidate
            if person is None:
                person = self._new_track(det, mot_id, current_time)
            det["matched_person"] = person
            active_track_ids.add(person.track_id)
            self._update_track(person, det, frame, frame_shape, current_time)
        return active_track_ids

    def _new_track(self, det, mot_id, current_time):
        track_id = mot_id if mot_id is not None else f"Guest_hash_{int(current_time * 1000)}_{self.track_id_counter}"
        self.track_id_counter += 1
        person = Person(track_id, np.zeros(128, dtype=np.float32), det["box"])
        person.name = UNKNOWN_NAME
        person.state = TrackState.UNREGISTERED
        person.next_state = TrackState.UNREGISTERED
        person.verification_status = Verification.UNKNOWN
        self.tracked_persons[track_id] = person
        return person

    def _update_track(self, person, det, frame, frame_shape, current_time):
        person.last_seen = current_time
        if person.verification_status == Verification.VERIFIED:
            # Presence, not primary status, keeps an identity's session alive.
            live_session = self.active_sessions.get(person.name)
            if live_session:
                live_session.last_seen = current_time
        dt = current_time - person.last_update if person.last_update else 0.033
        person.last_update = current_time
        person.box = det["box"]
        person.pose = det.get("pose", getattr(person, "pose", None))
        # Detections without a solvable face carry no angles; keep this track's own history.
        if det.get("pitch") is not None:
            person.pitch = (person.pitch * 0.6) + (det["pitch"] * 0.4)
            person.yaw = det["yaw"]
            person.roll = det["roll"]

        pose = person.pose
        if pose:
            self._smooth_pose_signals(person, pose)
            if person.track_id == self.primary_user_track_id and "l_eye" in det and "r_eye" in det:
                if settings.get("enable_eye_tracking", True):
                    gaze_x, is_looking_away = self.person_detector.extract_pupil_gaze(
                        frame, det["l_eye"], det["r_eye"], settings.gaze_away_threshold
                    )
                else:
                    gaze_x, is_looking_away = 0.0, False
                person.gaze_x = gaze_x
                person.is_looking_away = is_looking_away
                self._evaluate_single_target_health(
                    person, pose, person.current_ratio, dt, current_time, frame_shape
                )

        self._flush_analytics(person, current_time)

    @staticmethod
    def _smooth_pose_signals(person, pose):
        shoulder_width = np.abs(pose["left_shoulder"].x - pose["right_shoulder"].x)
        shoulder_center_y = (pose["left_shoulder"].y + pose["right_shoulder"].y) / 2.0
        current_ratio = np.abs(shoulder_center_y - pose["nose"].y) / max(shoulder_width, 1e-6)
        person.current_ratio = current_ratio
        person.last_y = shoulder_center_y

        alpha = 0.10
        if person.smoothed_pitch is None:
            person.smoothed_pitch = person.pitch
            person.smoothed_ratio = current_ratio
            person.smoothed_y = shoulder_center_y
        else:
            person.smoothed_pitch = (1 - alpha) * person.smoothed_pitch + alpha * person.pitch
            person.smoothed_ratio = (1 - alpha) * person.smoothed_ratio + alpha * current_ratio
            person.smoothed_y = (1 - alpha) * person.smoothed_y + alpha * shoulder_center_y

    def _flush_analytics(self, person, current_time):
        """Every 10 s, records the registered user's state (bystanders are never recorded)."""
        flush_interval = current_time - person.last_analytics_flush_time
        if flush_interval < _ANALYTICS_INTERVAL_SECONDS:
            return
        person.last_analytics_flush_time = current_time
        if person.verification_status != Verification.VERIFIED or person.name not in self.profiles:
            return
        session = self.active_sessions.get(person.name)
        self.db_manager.log_analytics_flush(
            user_name=person.name,
            session_state=str(person.state),
            duration_seconds=flush_interval,
            continuous_sitting_seconds=session.sitting_duration_clock if session else 0.0,
            continuous_gaze_seconds=session.screen_gaze_accumulation_timer if session else 0.0,
            average_head_pitch=person.smoothed_pitch if person.smoothed_pitch is not None else person.pitch,
            ocular_break_accumulator=session.ocular_break_timer if session else 0.0,
            spine_alignment=getattr(person, "spine_alignment", 0.0),
            shoulder_alignment=getattr(person, "shoulder_alignment", 0.0),
        )

    def _guard_primary_seat(self, active_track_ids):
        """A much larger (closer) unregistered person takes the seat from the verified user (anti-hijacking)."""
        active_tracks = [p for p in self.tracked_persons.values() if p.track_id in active_track_ids]
        verified_tracks = [
            p for p in active_tracks if p.verification_status == Verification.VERIFIED and p.state in ACTIVE_STATES
        ]
        unverified_tracks = [p for p in active_tracks if p.verification_status != Verification.VERIFIED]

        for v_track in verified_tracks:
            v_area = (v_track.box[2] - v_track.box[0]) * (v_track.box[3] - v_track.box[1])
            for u_track in unverified_tracks:
                u_area = (u_track.box[2] - u_track.box[0]) * (u_track.box[3] - u_track.box[1])
                if u_area > (v_area * 1.5):
                    v_track.state = TrackState.UNREGISTERED
                    v_track.next_state = TrackState.UNREGISTERED
                    v_track.name = UNKNOWN_NAME
                    v_track.verification_status = Verification.UNKNOWN
                    v_track.biometric_match_counter = 0

                    u_track.biometric_match_counter = 0
                    u_track.state = TrackState.VERIFYING
                    u_track.next_state = TrackState.VERIFYING

                    if self.primary_user_track_id == v_track.track_id:
                        self.primary_user_track_id = None
                        self.current_authenticated_user = None
                    break

    def _submit_biometric_job(self, frame, frame_shape, active_track_ids):
        """Sends one unverified person's head crop to the face worker, unless a verified user is active."""
        if not self.worker.is_idle() or any(
            p.verification_status == Verification.VERIFIED and p.state in ACTIVE_STATES
            for p in self.tracked_persons.values()
        ):
            return
        frame_h, frame_w = frame_shape[0], frame_shape[1]
        for track_id in active_track_ids:
            person = self.tracked_persons[track_id]
            if person.verification_status == Verification.VERIFIED:
                continue
            pose = getattr(person, "pose", None)
            if pose:
                x1, y1, x2, y2 = head_crop_box(pose, frame_w, frame_h)
            else:
                x1, y1 = int(max(0, person.box[0])), int(max(0, person.box[1]))
                x2, y2 = int(min(frame_w, person.box[2])), int(min(frame_h, person.box[3]))
            if x2 <= x1 or y2 <= y1:
                continue
            # Only the head crop crosses the process boundary, not the full frame.
            if self.worker.submit(person.track_id, frame[y1:y2, x1:x2].copy()):
                person.next_state = TrackState.VERIFYING
                person.verification_status = Verification.VERIFYING
                break

    # ------------------------------------------------------------------
    # 4. Track lifecycle and primary user
    # ------------------------------------------------------------------

    def _update_lifecycle(self, active_track_ids, current_time):
        expired = []
        for track_id, person in list(self.tracked_persons.items()):
            if track_id not in active_track_ids:
                time_since_seen = current_time - person.last_seen
                if time_since_seen > _EVICT_AFTER_SECONDS:
                    if person.name and person.name not in _NO_SESSION_METRICS:
                        session = self.active_sessions.get(person.name)
                        sitting_clock = session.sitting_duration_clock if session else 0.0
                        self.db_manager.log_session_metrics(person.name, "session_ended", sitting_clock)
                    expired.append(track_id)
                elif time_since_seen > _SEARCH_AFTER_SECONDS:
                    if person.state not in (TrackState.CALIBRATING, TrackState.ABSENT):
                        person.next_state = TrackState.SEARCHING
            elif person.verification_status == Verification.VERIFIED:
                if person.state in (TrackState.ABSENT, TrackState.SEARCHING):
                    person.next_state = TrackState.TRACKING
                    person.lost_grace_timer = 0.0
            elif person.verification_status == Verification.UNKNOWN:
                person.next_state = TrackState.UNREGISTERED

            self._commit_state(person, current_time)

        for track_id in expired:
            del self.tracked_persons[track_id]

        for name in [n for n, s in self.active_sessions.items() if current_time - s.last_seen > _SESSION_TTL_SECONDS]:
            del self.active_sessions[name]

    @staticmethod
    def _commit_state(person, current_time):
        """Moves to next_state once it has been requested for long enough (time-based, so FPS-independent)."""
        target = getattr(person, "next_state", person.state)
        if target != person.state:
            if getattr(person, "target_next_state", None) != target:
                person.target_next_state = target
                person.target_state_since = current_time
            if current_time - person.target_state_since >= _STATE_HOLD_SECONDS.get(target, 0.0):
                person.state = target
                person.target_next_state = None
        else:
            person.target_next_state = None

        if getattr(person, "_last_printed_state", None) != person.state:
            logger.debug(f"[STATE CHANGE] Track {person.track_id} -> {person.state}")
            person._last_printed_state = person.state
            person.last_state = person.state

    def _select_primary_user(self, frame_shape, current_time):
        """The verified, active person closest to the frame centre is the user being monitored."""
        frame_width = frame_shape[1]
        frame_cx = frame_width / 2.0

        closest_user = None
        closest_dist = float("inf")
        for p in self.tracked_persons.values():
            if p.state in ACTIVE_STATES and p.verification_status == Verification.VERIFIED:
                dist = abs((p.box[0] + p.box[2]) / 2.0 - frame_cx)
                if dist <= frame_width * settings.workspace_max_offset_ratio and dist < closest_dist:
                    closest_dist = dist
                    closest_user = p

        if closest_user:
            self.anchor_lost_since = None
            current_occupant = closest_user.name
        else:
            if self.anchor_lost_since is None:
                self.anchor_lost_since = current_time
            anchor_lost = current_time - self.anchor_lost_since > _ANCHOR_LOST_SECONDS
            current_occupant = None if anchor_lost else self.current_authenticated_user

        if current_occupant != self.last_session_owner:
            if current_occupant is not None:
                logger.info(f"[SESSION] Primary seat occupied by: {current_occupant}")
            else:
                logger.info("[SESSION] Primary seat vacated.")
            self.last_session_owner = current_occupant

        if closest_user:
            self.current_authenticated_user = closest_user.name
            self.primary_user_track_id = closest_user.track_id
        else:
            if current_occupant is None:
                self.current_authenticated_user = None
            self.primary_user_track_id = None
