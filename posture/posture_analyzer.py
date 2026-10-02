import numpy as np
from typing import Any, Dict, Tuple
from config.settings_manager import settings
from core.tracking.state import Person
from core.tracking.geometry import Keypoint


def resolve_slouch_sensitivity(person, session) -> float:
    """Per-user neck tolerance (degrees): the profile value on the Person wins over defaults."""
    for source in (person, session):
        value = getattr(source, "slouch_sensitivity", None)
        if value is not None:
            return float(value)
    return float(settings.get("neck_pitch_tolerance_degrees", 15.0))


class PostureState:
    def __init__(self):
        self.is_slouching = False
        self.is_standing = False
        self.neck_pitch = 0.0
        self.spine_alignment = 0.0
        self.shoulder_alignment = 0.0


class PostureAnalyzer:
    """
    Stateless evaluator for posture analytics.
    Computes human skeleton geometric bounds and evaluates them against the user's calibrated baseline.
    """

    @staticmethod
    def _compute_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Calculates the 3D angle between points A, B, and C with B as the vertex."""
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    @staticmethod
    def _compute_shoulder_alignment(
        left_shoulder: Keypoint, right_shoulder: Keypoint
    ) -> float:
        """Shoulder-line tilt from horizontal in pixel space, in (-90, 90]; positive = subject's left shoulder lower."""
        dx = left_shoulder.px - right_shoulder.px
        dy = left_shoulder.py - right_shoulder.py
        # Facing the camera the subject's left shoulder is on the image right; facing away it is not.
        if dx < 0:
            dx, dy = -dx, -dy
        return np.degrees(np.arctan2(dy, dx))

    @staticmethod
    def _compute_spine_alignment(
        shoulder_center: Tuple[float, float], hip_center: Tuple[float, float]
    ) -> float:
        """Spine angle from vertical, in degrees; expects pixel coordinates."""
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        angle = np.degrees(np.arctan2(dy, dx))
        # Perfect vertical spine is -90 degrees in OpenCV coordinate space (y goes down).
        return abs(angle + 90.0)

    @staticmethod
    def evaluate(
        person: Person, session: "Any", pose: Dict[str, Keypoint], frame_shape: Tuple[int, int]
    ) -> PostureState:
        """
        Evaluates the current skeletal pose against the session's baseline to determine slouching and standing state.
        """
        state = PostureState()

        l_shoulder = pose.get("left_shoulder")
        r_shoulder = pose.get("right_shoulder")
        l_hip = pose.get("left_hip")
        r_hip = pose.get("right_hip")
        nose = pose.get("nose")

        if not all([l_shoulder, r_shoulder, nose]):
            return state

        shoulder_width = np.abs(r_shoulder.x - l_shoulder.x)
        shoulder_center_y = (l_shoulder.y + r_shoulder.y) / 2.0

        # 1. Torso Depth Ratio (Z-axis leaning surrogate)
        current_torso_depth_ratio = np.abs(shoulder_center_y - nose.y) / max(
            shoulder_width, 1e-6
        )

        # 2. Neck Pitch (from PnP smoothed roll/pitch/yaw already on person)
        current_neck_pitch_angle = (
            person.smoothed_pitch
            if getattr(person, "smoothed_pitch", None) is not None
            else person.pitch
        )
        state.neck_pitch = current_neck_pitch_angle

        if getattr(session, "calibrated_baseline_neck_pitch", None) in (None, 0.0):
            session.calibrated_baseline_neck_pitch = current_neck_pitch_angle

        relative_slouch = (
            current_neck_pitch_angle - session.calibrated_baseline_neck_pitch
        )

        if getattr(session, "posture_baseline", None) is None:
            session.posture_baseline = 0.5

        # 3. Shoulder and Spine Alignment
        state.shoulder_alignment = PostureAnalyzer._compute_shoulder_alignment(
            l_shoulder, r_shoulder
        )
        if l_hip and r_hip:
            state.spine_alignment = PostureAnalyzer._compute_spine_alignment(
                ((l_shoulder.px + r_shoulder.px) / 2.0, (l_shoulder.py + r_shoulder.py) / 2.0),
                ((l_hip.px + r_hip.px) / 2.0, (l_hip.py + r_hip.py) / 2.0),
            )

        # 4. Slouch Evaluation
        slouch_sensitivity = resolve_slouch_sensitivity(person, session)
        if session.health_status == "Posture Deficit Alert":
            # Must satisfy a stricter upright bound to clear an existing alert
            is_fully_upright = (
                current_torso_depth_ratio >= session.posture_baseline * 0.95
            ) and (relative_slouch <= slouch_sensitivity * 0.40)
            state.is_slouching = not is_fully_upright
        else:
            # Tolerant bounds to trigger a new alert
            is_spine_skewed = state.spine_alignment > settings.spine_tolerance_degrees
            state.is_slouching = (
                (current_torso_depth_ratio < (session.posture_baseline * settings.slouch_ratio_threshold))
                or (relative_slouch > slouch_sensitivity)
                or is_spine_skewed
            )

        # 5. Standing Evaluation
        baseline_shoulder_y = getattr(session, "baseline_shoulder_y", None)
        if baseline_shoulder_y is None:
            baseline_shoulder_y = shoulder_center_y
        normalized_height_delta = (baseline_shoulder_y - shoulder_center_y) / max(shoulder_width, 1e-6)
        
        box_h = person.box[3] - person.box[1]
        box_w = person.box[2] - person.box[0]
        box_ratio = box_h / max(box_w, 1.0)
        
        # A true stand causes a massive vertical shift or a significantly taller bounding box.
        # We use strict thresholds to prevent simple forward leaning from triggering this.
        state.is_standing = (normalized_height_delta > 0.85) or (
            current_torso_depth_ratio > (session.posture_baseline * 2.2) and box_ratio > 1.8
        )

        return state
