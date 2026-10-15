# Default application settings

DEFAULT_SETTINGS = {
    # Biometric Tracking Tolerances (ArcFace)
    "recognition_threshold_initial": 0.25,
    "ambiguity_margin": 0.05,
    "confirmation_frame_count": 5,

    # Face Quality Bounds
    "face_min_width": 60,
    "min_blur_laplacian": 100.0,

    # Spatial Tracking Bounds
    "workspace_min_area_ratio": 0.15,
    "workspace_max_offset_ratio": 0.40,

    # Posture Thresholds
    "spine_tolerance_degrees": 20.0,
    "neck_pitch_tolerance_degrees": 15.0,  # default per-user slouch sensitivity for new profiles
    "shoulder_roll_tolerance_degrees": 10.0,
    "slouch_sensitivity": "Medium",  # Low, Medium, High

    # Eye Tracking Settings
    "enable_eye_tracking": True,
    "screen_gaze_limit_seconds": 1200,
    "ocular_break_duration_seconds": 20,
    "eye_tracking_sensitivity": "Medium",  # Low, Medium, High

    # Sitting & Movement
    "session_limit_seconds": 1200,
    "stand_requirement_seconds": 180,
    "enable_break_reminders": True,

    # Notifications
    "enable_sound_alerts": True,
    "reminder_cooldown": 300,  # seconds

    # Detection
    "detection_fps": 30,

    # Data retention
    "analytics_retention_days": 90,

    # Camera Subsystem
    "camera": {
        "preferred_camera": "",
        "resolution": [1280, 720],
        "fps": 30,
        "buffer_size": 1,
    },
}
