"""Validation for data entering through the HTTP API."""
import math
import unicodedata
from urllib.parse import quote

from config.defaults import DEFAULT_SETTINGS

MAX_NAME_LENGTH = 40
_NAME_PUNCTUATION = set(" .-'\u2019")
_LEVELS = {"Low", "Medium", "High"}

SETTINGS_SCHEMA = {
    "recognition_threshold_initial": ("float", 0.0, 1.0),
    "ambiguity_margin": ("float", 0.0, 1.0),
    "confirmation_frame_count": ("int", 1, 50),
    "face_min_width": ("int", 16, 1024),
    "min_blur_laplacian": ("float", 0.0, 10000.0),
    "workspace_min_area_ratio": ("float", 0.0, 1.0),
    "workspace_max_offset_ratio": ("float", 0.0, 1.0),
    "spine_tolerance_degrees": ("float", 1.0, 90.0),
    "neck_pitch_tolerance_degrees": ("float", 1.0, 90.0),
    "shoulder_roll_tolerance_degrees": ("float", 1.0, 90.0),
    "slouch_sensitivity": ("enum", _LEVELS),
    "enable_eye_tracking": ("bool",),
    "screen_gaze_limit_seconds": ("int", 60, 14400),
    "ocular_break_duration_seconds": ("int", 5, 600),
    "eye_tracking_sensitivity": ("enum", _LEVELS),
    "session_limit_seconds": ("int", 300, 14400),
    "stand_requirement_seconds": ("int", 30, 3600),
    "enable_break_reminders": ("bool",),
    "enable_sound_alerts": ("bool",),
    "reminder_cooldown": ("int", 10, 3600),
    # Posture calibration averages a 3 s window and needs at least 5 samples.
    "detection_fps": ("int", 5, 60),
    "analytics_retention_days": ("int", 1, 3650),
}

# cached_pipeline_* entries are deliberately absent: they are written only by CameraManager.
CAMERA_SCHEMA = {
    "preferred_camera": ("camera_index",),
    "resolution": ("resolution",),
    "fps": ("int", 1, 120),
    "buffer_size": ("int", 1, 10),
}

PROFILE_SCHEMA = {
    "slouch_sensitivity": ("float", 1.0, 90.0),
    "session_limit": ("int", 60, 14400),
    "stand_requirement": ("int", 10, 3600),
    "screen_gaze_limit": ("int", 60, 14400),
    # The old dashboard stored up to 15 minutes here; keep those profiles editable.
    "ocular_break_duration": ("int", 5, 900),
}


def validate_profile_name(value) -> str:
    if not isinstance(value, str):
        raise ValueError("Profile name must be text.")
    name = unicodedata.normalize("NFC", value).strip()
    if not 1 <= len(name) <= MAX_NAME_LENGTH:
        raise ValueError(f"Profile name must be 1-{MAX_NAME_LENGTH} characters.")
    # The tracker treats any name starting with "Unknown" as an unregistered person.
    if name.casefold().startswith("unknown"):
        raise ValueError("Profile name cannot start with 'Unknown'.")
    if not unicodedata.category(name[0]).startswith(("L", "N")):
        raise ValueError("Profile name must start with a letter or digit.")
    for ch in name:
        if ch not in _NAME_PUNCTUATION and not unicodedata.category(ch).startswith(("L", "M", "N")):
            raise ValueError("Profile name may only contain letters, digits, spaces and . - ' characters.")
    return name


def validate_profile_key(value) -> str:
    """Names an existing profile; looser than validate_profile_name so profiles created before it stay manageable."""
    if not isinstance(value, str) or not 1 <= len(value) <= 200:
        raise ValueError("Profile name is required.")
    return value


def _number(value, kind, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"must be between {low} and {high}")
    if kind == "int":
        if value != int(value):
            raise ValueError("must be a whole number")
        value = int(value)
    else:
        value = float(value)
    if not low <= value <= high:
        raise ValueError(f"must be between {low} and {high}")
    return value


def _coerce(value, spec):
    kind = spec[0]
    if kind in ("int", "float"):
        return _number(value, kind, spec[1], spec[2])
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError("must be true or false")
        return value
    if kind == "enum":
        if not isinstance(value, str) or value not in spec[1]:
            raise ValueError(f"must be one of {sorted(spec[1])}")
        return value
    if kind == "camera_index":
        if value == "":
            return value
        return _number(value, "int", 0, 63)
    if kind == "resolution":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("must be [width, height]")
        return [_number(value[0], "int", 64, 7680), _number(value[1], "int", 64, 4320)]
    raise ValueError("unsupported setting")


def sanitize_settings(payload) -> dict:
    """Returns only known, well-typed settings. Unknown keys are dropped; invalid values raise."""
    if not isinstance(payload, dict):
        raise ValueError("Settings payload must be a JSON object.")
    clean, errors = {}, []
    for key, value in payload.items():
        if key == "camera":
            if not isinstance(value, dict):
                errors.append("camera: must be an object")
                continue
            camera = {}
            for sub_key, sub_value in value.items():
                spec = CAMERA_SCHEMA.get(sub_key)
                if spec is None:
                    continue
                try:
                    camera[sub_key] = _coerce(sub_value, spec)
                except ValueError as e:
                    errors.append(f"camera.{sub_key}: {e}")
            if camera:
                clean["camera"] = camera
            continue
        spec = SETTINGS_SCHEMA.get(key)
        if spec is None:
            continue
        try:
            clean[key] = _coerce(value, spec)
        except ValueError as e:
            errors.append(f"{key}: {e}")
    if errors:
        raise ValueError("; ".join(errors))
    return clean


def validate_settings_keys(keys) -> list:
    if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
        raise ValueError("keys must be a list of setting names.")
    return [k for k in keys if k in DEFAULT_SETTINGS]


def validate_profile_update(data) -> dict:
    """Name plus the limits being changed; untouched limits are not revalidated, so legacy values survive."""
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")
    clean, errors = {"name": validate_profile_key(data.get("name"))}, []
    for key, spec in PROFILE_SCHEMA.items():
        if key not in data:
            continue
        try:
            clean[key] = _coerce(data[key], spec)
        except ValueError as e:
            errors.append(f"{key}: {e}")
    if errors:
        raise ValueError("; ".join(errors))
    if len(clean) == 1:
        raise ValueError(f"Nothing to update; expected one of {sorted(PROFILE_SCHEMA)}.")
    return clean


def attachment_header(filename: str) -> str:
    """Content-Disposition value that is safe for any profile name."""
    ascii_name = "".join(c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in filename)
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"
