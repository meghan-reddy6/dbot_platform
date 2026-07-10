import numpy as np
from typing import Dict, Tuple
from core.tracking.state import Person
from core.tracking.geometry import Keypoint


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
        """Calculates the roll angle between shoulders (leaning left/right)."""
        dx = right_shoulder.x - left_shoulder.x
        dy = right_shoulder.y - left_shoulder.y
        return np.degrees(np.arctan2(dy, dx))

    @staticmethod
    def _compute_spine_alignment(
        shoulder_center: Tuple[float, float], hip_center: Tuple[float, float]
    ) -> float:
        """Calculates the angle of the spine relative to the vertical axis (leaning left/right)."""
        dx = shoulder_center[0] - hip_center[0]
        dy = shoulder_center[1] - hip_center[1]
        angle = np.degrees(np.arctan2(dy, dx))
        # Perfect vertical spine is -90 degrees in OpenCV coordinate space (y goes down).
        # We normalize to 0 being perfectly vertical.
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
        shoulder_center_x = (l_shoulder.x + r_shoulder.x) / 2.0
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

        if getattr(session, "calibrated_baseline_neck_pitch", 0.0) == 0.0:
            session.calibrated_baseline_neck_pitch = current_neck_pitch_angle

        relative_slouch = (
            current_neck_pitch_angle - session.calibrated_baseline_neck_pitch
        )

        if not hasattr(session, "posture_baseline"):
            session.posture_baseline = 0.5

        # 3. Shoulder and Spine Alignment
        state.shoulder_alignment = PostureAnalyzer._compute_shoulder_alignment(
            l_shoulder, r_shoulder
        )
        if l_hip and r_hip:
            hip_center_x = (l_hip.x + r_hip.x) / 2.0
            hip_center_y = (l_hip.y + r_hip.y) / 2.0
            state.spine_alignment = PostureAnalyzer._compute_spine_alignment(
                (shoulder_center_x, shoulder_center_y), (hip_center_x, hip_center_y)
            )

        # 4. Slouch Evaluation
        if session.health_status == "Posture Deficit Alert":
            # Must satisfy a stricter upright bound to clear an existing alert
            is_fully_upright = (
                current_torso_depth_ratio >= session.posture_baseline * 0.95
            ) and (relative_slouch <= getattr(session, "slouch_sensitivity", 15.0) * 0.40)
            state.is_slouching = not is_fully_upright
        else:
            # Tolerant bounds to trigger a new alert
            is_spine_skewed = (
                state.spine_alignment > 15.0
            )  # Leaning sideways > 15 degrees
            state.is_slouching = (
                (current_torso_depth_ratio < (session.posture_baseline * 0.80))
                or (relative_slouch > 35.0)
                or is_spine_skewed
            )

        # 5. Standing Evaluation
        normalized_height_delta = (
            getattr(session, "baseline_shoulder_y", shoulder_center_y)
            - shoulder_center_y
        ) / max(shoulder_width, 1e-6)
        
        box_h = person.box[3] - person.box[1]
        box_w = person.box[2] - person.box[0]
        box_ratio = box_h / max(box_w, 1.0)
        
        # A true stand causes a massive vertical shift or a significantly taller bounding box.
        # We use strict thresholds to prevent simple forward leaning from triggering this.
        state.is_standing = (normalized_height_delta > 0.85) or (
            current_torso_depth_ratio > (session.posture_baseline * 2.2) and box_ratio > 1.8
        )

        return state
