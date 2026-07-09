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

    # Biometric Tracking Tolerances
    admission_threshold_strict: float = 0.910
    admission_threshold_legacy: float = 0.880
    hysteresis_holding_threshold: float = 0.850

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
