"""Calibration, posture, screen-time and sitting timers for the primary user, and their spoken reminders."""
import logging
import time

import numpy as np

from config.settings_manager import settings
from posture.correction_engine import CorrectionEngine
from posture.posture_analyzer import PostureAnalyzer

from .state import UserSession
from .states import HealthStatus, TrackState

logger = logging.getLogger(__name__)

_CALIBRATION_SECONDS = 3.0
_RECOVERY_CALIBRATION_SECONDS = 1.0
_STANDING_CONFIRM_SECONDS = 2.5
_SLOUCH_CONFIRM_SECONDS = 4.0
_UPRIGHT_CONFIRM_SECONDS = 1.5
_HEALTH_STATES = (TrackState.TRACKING, TrackState.STANDING, TrackState.LOOKING_AWAY, TrackState.SEARCHING)


def _torso_ratio(pose):
    """(nose-to-shoulder height / shoulder width, shoulder-centre y), both in normalised coordinates."""
    shoulder_width = np.abs(pose["right_shoulder"].x - pose["left_shoulder"].x)
    shoulder_center_y = (pose["left_shoulder"].y + pose["right_shoulder"].y) / 2.0
    return np.abs(shoulder_center_y - pose["nose"].y) / max(shoulder_width, 1e-6), shoulder_center_y


class HealthMonitorMixin:
    """TrackerEngine's per-identity health logic; it shares the engine's sessions and primary-user state."""

    def _dispatch_voice(
        self, text: str, category: str = "general", cooldown: float = 10.0, identity: str = "Unknown"
    ) -> None:
        """Dispatches an asynchronous voice alert through the dedicated daemon queue."""
        if hasattr(self, "alert_manager"):
            self.last_system_message = text
            self.alert_manager.dispatch(text, category, cooldown, identity)

    def _restore_session(self, person, current_time):
        """Rebuilds a UserSession from a live, calibrated track whose session was garbage-collected."""
        session = UserSession(person.name)
        session.posture_baseline = person.posture_baseline
        session.calibrated_baseline_neck_pitch = getattr(person, "calibrated_baseline_neck_pitch", None)
        session.baseline_shoulder_y = getattr(person, "baseline_shoulder_y", None)
        session.is_posture_calibrated = True
        session.last_seen = current_time
        self.active_sessions[person.name] = session
        logger.info(f"[SESSION] Restored session for {person.name} from its live track.")
        return session

    def _evaluate_single_target_health(
        self,
        person,
        pose: dict,
        current_ratio: float,
        dt: float,
        current_time: float,
        frame_shape: tuple,
    ) -> None:
        """Advances calibration or the health timers of one tracked person by one frame."""
        if (
            person.name.startswith("Unknown") or self.primary_user_track_id is None
        ) and person.state != TrackState.CALIBRATING:
            person.next_state = TrackState.UNREGISTERED_TARGET
            person.sustained_slouch_debounce_timer = 0.0
            person.tracking_active_debounce_timer = 0.0
            return

        if person.recovery_calibration_start is not None:
            self._recovery_calibration_step(person, pose, current_time)
            return

        if person.state == TrackState.CALIBRATING:
            # Calibration blocks every downstream evaluation until it completes.
            self._calibration_step(person, pose, current_time)
            return

        session = self.active_sessions.get(person.name)
        if session:
            session.last_seen = current_time
        elif getattr(person, "posture_baseline", None) is not None:
            session = self._restore_session(person, current_time)
        else:
            person.next_state = TrackState.CALIBRATING
            return

        if person.state in _HEALTH_STATES:
            self._update_health(person, session, pose, current_ratio, dt, current_time, frame_shape)

    def _recovery_calibration_step(self, person, pose, current_time):
        person.recovery_accumulator.append(_torso_ratio(pose))
        if current_time - person.recovery_calibration_start < _RECOVERY_CALIBRATION_SECONDS:
            return
        if person.recovery_accumulator:
            person.posture_baseline = float(np.mean([i[0] for i in person.recovery_accumulator]))
            person.baseline_shoulder_y = float(np.mean([i[1] for i in person.recovery_accumulator]))
        person.recovery_calibration_start = None
        person.recovery_accumulator = []

    def _calibration_step(self, person, pose, current_time):
        if getattr(person, "calibration_start_time", None) is None:
            person.calibration_start_time = current_time
            person.calibration_accumulator = []
            person.calibration_pitch_acc = []
            person.calibration_y_acc = []
            person.is_posture_calibrated = False

        if not getattr(person, "calibration_announced", False):
            person.calibration_announced = True
            self._dispatch_voice(
                f"{person.name}, please look straight ahead to calibrate your posture baseline.",
                category="calibration_prompt",
                identity=person.name,
            )

        ratio, shoulder_center_y = _torso_ratio(pose)
        if current_time - getattr(person, "last_log_time", 0.0) >= 1.0:
            person.last_log_time = current_time
        for attribute in ("calibration_accumulator", "calibration_pitch_acc", "calibration_y_acc"):
            if not hasattr(person, attribute):
                setattr(person, attribute, [])
        person.calibration_accumulator.append(ratio)
        person.calibration_pitch_acc.append(person.pitch)
        person.calibration_y_acc.append(shoulder_center_y)

        if current_time - person.calibration_start_time < _CALIBRATION_SECONDS:
            return

        if len(person.calibration_accumulator) >= 5:
            person.posture_baseline = float(np.mean(person.calibration_accumulator))
            person.calibrated_baseline_neck_pitch = float(np.mean(person.calibration_pitch_acc))
            person.baseline_shoulder_y = float(np.mean(person.calibration_y_acc))
        else:
            if not hasattr(person, "posture_baseline"):
                person.calibrated_baseline_neck_pitch = 0.0
                person.posture_baseline = 0.50
                person.baseline_shoulder_y = (
                    float(np.mean(person.calibration_y_acc)) if person.calibration_y_acc else 0.50
                )
            logger.info(
                f"[CALIBRATION] Timeout fallback triggered. Preserving/Forcing track active for {person.name}."
            )
        person.is_posture_calibrated = True

        if person.name not in self.active_sessions:
            self.active_sessions[person.name] = UserSession(person.name)
        session = self.active_sessions[person.name]
        session.posture_baseline = person.posture_baseline
        session.calibrated_baseline_neck_pitch = person.calibrated_baseline_neck_pitch
        session.baseline_shoulder_y = person.baseline_shoulder_y
        session.is_posture_calibrated = True
        session.calibrated_at = time.time()
        session.last_seen = current_time

        person.calibration_start_time = None
        person.calibration_accumulator = []
        person.calibration_pitch_acc = []
        person.calibration_y_acc = []
        person.next_state = TrackState.TRACKING

        self.primary_user_track_id = person.track_id
        self._dispatch_voice(
            f"Calibration successful for {person.name}. Posture monitoring is now active.",
            "calibration_success",
            30.0,
            identity=person.name,
        )

    def _update_health(self, person, session, pose, current_ratio, dt, current_time, frame_shape):
        """The order of these steps matters: later ones read the health status earlier ones set."""
        eye_tracking = settings.get("enable_eye_tracking", True)
        looking_away = person.is_looking_away or person.state == TrackState.SEARCHING

        if eye_tracking and not looking_away:
            session.screen_gaze_accumulation_timer += dt
            if session.health_status != HealthStatus.OCULAR_BREAK:
                session.ocular_break_timer = 0.0

        is_standing, shoulder_center_y = self._classify_posture(person, session, pose, dt, frame_shape)
        self._check_eye_break(person, session, dt, eye_tracking, looking_away)
        if person.state != TrackState.STANDING:
            self._check_sitting_limit(person, session, dt)
        self._announce_posture(person, session, dt)
        if person.state == TrackState.STANDING:
            self._track_standing_break(person, session, dt, is_standing, shoulder_center_y)
        if person.state == TrackState.TRACKING:
            self._drift_baselines(person, session, shoulder_center_y, current_ratio)

        # Sync back to person for drawing loop to read
        person.health_status = session.health_status

        if current_time - getattr(session, "last_debug_log_time", 0) > 5.0:
            logger.debug(
                f"\n[POSTURE DEBUG]\n"
                f"Identity: {session.identity_name}\n"
                f"Track: {person.track_id}\n"
                f"Session Age: {int(current_time - session.first_seen_time)}s\n"
                f"Sitting Duration: {int(session.sitting_duration_clock)}s\n"
                f"Current Posture: {'Standing' if session.standing_accumulator_time > 0 else 'Slouching' if session.slouch_accumulator_time > 0 else 'Active'}\n"
                f"Slouch Duration: {int(session.slouch_accumulator_time)}s\n"
            )
            session.last_debug_log_time = current_time

    def _classify_posture(self, person, session, pose, dt, frame_shape):
        shoulder_center_y = (pose["left_shoulder"].y + pose["right_shoulder"].y) / 2.0
        posture_state = PostureAnalyzer.evaluate(person, session, pose, frame_shape)
        is_standing = posture_state.is_standing
        person.spine_alignment = posture_state.spine_alignment
        person.shoulder_alignment = posture_state.shoulder_alignment
        person.is_standing = is_standing

        is_pinned_to_ceiling = person.box[1] <= frame_shape[0] * 0.05
        nose_missing = pose["nose"].y <= 0.01
        is_standing_frame = is_standing or (is_pinned_to_ceiling and nose_missing)
        is_slouching_frame = posture_state.is_slouching and not is_standing_frame

        if is_standing_frame:
            session.standing_accumulator_time += dt
            session.slouch_accumulator_time = 0.0
            session.active_accumulator_time = 0.0
        elif is_slouching_frame:
            session.slouch_accumulator_time += dt
            session.standing_accumulator_time = 0.0
            session.active_accumulator_time = 0.0
        else:
            session.active_accumulator_time += dt
            session.standing_accumulator_time = 0.0
            session.slouch_accumulator_time = 0.0

        if session.standing_accumulator_time >= _STANDING_CONFIRM_SECONDS:
            person.next_state = TrackState.STANDING
        elif session.slouch_accumulator_time >= _SLOUCH_CONFIRM_SECONDS:
            session.health_status = HealthStatus.POSTURE_DEFICIT
        elif session.active_accumulator_time >= _UPRIGHT_CONFIRM_SECONDS:
            if person.state == TrackState.STANDING:
                person.next_state = TrackState.TRACKING
            if session.health_status == HealthStatus.POSTURE_DEFICIT:
                session.health_status = HealthStatus.HEALTHY
        return is_standing, shoulder_center_y

    def _check_eye_break(self, person, session, dt, eye_tracking, looking_away):
        if eye_tracking and session.screen_gaze_accumulation_timer >= getattr(person, "screen_gaze_limit", 1200):
            session.health_status = HealthStatus.OCULAR_BREAK
            if not session.ocular_break_announced:
                logger.info(f"[TIMER ALERT] Triggering Voice Alert for: Ocular Break Recommended ({person.name})")
                self._dispatch_voice(
                    f"{person.name}, attention, eye strain warning. Please look away from the screen.",
                    category="ocular_break",
                    identity=person.name,
                )
                session.ocular_break_announced = True

            if looking_away:
                session.ocular_break_timer += dt
                if session.ocular_break_timer >= getattr(person, "gaze_away_limit", 20):
                    session.screen_gaze_accumulation_timer = 0.0
                    session.ocular_break_timer = 0.0
                    session.health_status = HealthStatus.HEALTHY
                    session.ocular_break_announced = False
        elif session.health_status == HealthStatus.OCULAR_BREAK:
            session.health_status = HealthStatus.HEALTHY
            session.ocular_break_announced = False
            session.ocular_break_timer = 0.0

    def _check_sitting_limit(self, person, session, dt):
        session.sitting_duration_clock += dt
        if session.sitting_duration_clock >= getattr(person, "session_limit", 2400):
            session.health_status = HealthStatus.SESSION_LIMIT
            if not session.session_limit_announced and settings.get("enable_break_reminders", True):
                logger.info(f"[TIMER ALERT] Triggering Voice Alert for: Session Limit Reached ({person.name})")
                self._dispatch_voice(
                    f"{person.name}, you have been sitting for too long. Please stand up.",
                    category="session_limit",
                    identity=person.name,
                )
                session.session_limit_announced = True
        elif session.health_status == HealthStatus.SESSION_LIMIT:
            session.health_status = HealthStatus.HEALTHY
            session.session_limit_announced = False

    def _announce_posture(self, person, session, dt):
        if session.health_status == HealthStatus.POSTURE_DEFICIT:
            if not session.slouch_announced:
                advice = CorrectionEngine.get_advice(person, session)
                logger.info(
                    f"[TIMER ALERT] Triggering Voice Alert for: Posture Deficit Alert - {advice} ({person.name})"
                )
                self._dispatch_voice(
                    f"{advice} {person.name}.", category="posture_alert", cooldown=30.0, identity=person.name
                )
                session.slouch_announced = True
        else:
            session.slouch_timer = max(0.0, getattr(session, "slouch_timer", 0.0) - dt)
            session.slouch_announced = False

    def _track_standing_break(self, person, session, dt, is_standing, shoulder_center_y):
        session.standing_duration_clock += dt
        session.slouch_timer = max(0.0, getattr(session, "slouch_timer", 0.0) - dt)
        if session.standing_duration_clock >= getattr(person, "stand_requirement", 120):
            session.sitting_duration_clock = 0.0
            session.session_limit_announced = False
            if "Session Limit Reached" in getattr(session, "health_status", ""):
                session.health_status = HealthStatus.HEALTHY
        if is_standing and session.baseline_shoulder_y is not None:
            session.baseline_shoulder_y = (session.baseline_shoulder_y * 0.95) + (shoulder_center_y * 0.05)

    def _drift_baselines(self, person, session, shoulder_center_y, current_ratio):
        """Slowly follows the user's natural posture so the baseline does not go stale over a long session."""
        fast_frames = getattr(person, "fast_relatch_frames", 0)
        alpha = 0.20 if fast_frames > 0 else 0.005
        if session.baseline_shoulder_y is not None:
            session.baseline_shoulder_y = (session.baseline_shoulder_y * (1.0 - alpha)) + (shoulder_center_y * alpha)
        if session.posture_baseline is not None:
            session.posture_baseline = (session.posture_baseline * (1.0 - alpha)) + (current_ratio * alpha)
        if fast_frames > 0:
            person.fast_relatch_frames = fast_frames - 1
