# Default application settings

DEFAULT_SETTINGS = {
    # System Paths
    "model_base_dir": "D:\\Thundersoft\\dbot\\models",
    "profiles_cache_path": "profiles_cache.json",
    "database_path": "wellness_logs.db",
    
    # Biometric Tracking Tolerances (ArcFace)
    "recognition_threshold_initial": 0.25,
    "ambiguity_margin": 0.05,
    "confirmation_frame_count": 5,
    
    # Face Quality Bounds
    "face_min_width": 60,
    "min_blur_laplacian": 100.0,
    "max_yaw_pitch_degrees": 45.0,
    
    # Spatial Tracking Bounds
    "workspace_min_area_ratio": 0.15,
    "workspace_max_offset_ratio": 0.40,
    
    # Posture Thresholds
    "spine_tolerance_degrees": 20.0,
    "neck_pitch_tolerance_degrees": 15.0,
    "shoulder_roll_tolerance_degrees": 10.0,
    "slouch_sensitivity": "Medium", # Low, Medium, High
    "head_forward_threshold": 10.0,
    
    # Eye Tracking Settings
    "enable_eye_tracking": True,
    "screen_gaze_limit_seconds": 1200,
    "ocular_break_duration_seconds": 20,
    "eye_tracking_sensitivity": "Medium", # Low, Medium, High
    
    # Sitting & Movement
    "session_limit_seconds": 1200,
    "stand_requirement_seconds": 180,
    "enable_break_reminders": True,
    
    # Notifications
    "enable_desktop_notifications": True,
    "enable_sound_alerts": True,
    "enable_popup_alerts": True,
    "reminder_cooldown": 300, # seconds
    
    # Detection Sensitivity
    "confidence_threshold": 70, # percentage
    "tracking_stability": "Normal",
    "detection_fps": 30,
    "enable_pose_smoothing": True,
    
    # Camera Subsystem
    "camera": {
        "preferred_camera": "",
        "backend": "auto",
        "resolution": [1280, 720],
        "fps": 30,
        "buffer_size": 1,
        "auto_detect": True,
        "low_latency": True,
        "drop_old_frames": True,
        "hardware_acceleration": True
    },
}
