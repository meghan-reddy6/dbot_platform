from dataclasses import dataclass


@dataclass
class EdgeConfig:
    """
    Centralized configuration registry for the DeskBot V3 Edge AI Pipeline.
    """

    # System Paths
    model_base_dir: str = "D:\\Thundersoft\\dbot\\models"
    profiles_cache_path: str = "profiles_cache.json"
    database_path: str = "wellness_logs.db"

    # Biometric Tracking Tolerances (ArcFace)
    recognition_threshold_initial: float = 0.55
    ambiguity_margin: float = 0.15
    confirmation_frame_count: int = 5
    
    # Face Quality Bounds
    face_min_width: int = 60
    min_blur_laplacian: float = 100.0
    max_yaw_pitch_degrees: float = 45.0

    # Spatial Tracking Bounds
    workspace_min_area_ratio: float = 0.15
    workspace_max_offset_ratio: float = 0.40

    # Alerting & Posture Tolerances
    session_limit_seconds: int = 1200
    stand_requirement_seconds: int = 180
    screen_gaze_limit_seconds: int = 600

    spine_tolerance_degrees: float = 20.0
    neck_pitch_tolerance_degrees: float = 35.0
    shoulder_roll_tolerance_degrees: float = 15.0
    slouch_ratio_threshold: float = 0.80


# Instantiate a global config singleton
settings = EdgeConfig()
