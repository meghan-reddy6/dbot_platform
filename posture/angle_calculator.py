import math
import numpy as np


def calculate_angle_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculates the 3D angle between points A, B, and C with B as the vertex."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def compute_neck_pitch(shoulder_center: np.ndarray, nose: np.ndarray) -> float:
    """Calculates forward head pitch relative to the shoulders."""
    dx = nose[0] - shoulder_center[0]
    dy = nose[1] - shoulder_center[1]
    dz = nose[2] - shoulder_center[2] if len(nose) > 2 else 0.0

    pitch = math.degrees(math.atan2(dz, math.sqrt(dx * dx + dy * dy)))
    return pitch
