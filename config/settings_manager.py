import copy
import json
import logging
import os
import tempfile
import threading
from typing import Dict, Any, Optional
from config.defaults import DEFAULT_SETTINGS
from config.paths import data_path

logger = logging.getLogger(__name__)

_SLOUCH_RATIO = {"Low": 0.80, "Medium": 0.85, "High": 0.90}
_GAZE_AWAY_THRESHOLD = {"Low": 0.90, "Medium": 0.75, "High": 0.60}


class SettingsManager:
    """
    Manages loading, saving, and merging configuration overrides.
    Persists only values that differ from DEFAULT_SETTINGS.
    """

    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath or data_path("settings.json")
        self.lock = threading.RLock()
        self._current_settings = copy.deepcopy(DEFAULT_SETTINGS)
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
        self.load()

    def load(self) -> None:
        """Loads overrides from disk and merges them over the defaults."""
        with self.lock:
            merged = copy.deepcopy(DEFAULT_SETTINGS)
            if os.path.exists(self.filepath):
                try:
                    with open(self.filepath, "r", encoding="utf-8") as f:
                        overrides = self._valid_overrides(json.load(f))
                    for k, v in overrides.items():
                        if isinstance(merged[k], dict):
                            merged[k].update(v)
                        else:
                            merged[k] = v
                except (OSError, ValueError, TypeError, AttributeError, RecursionError) as e:
                    logger.error(f"Failed to load {self.filepath}, using defaults: {e}")
                    merged = copy.deepcopy(DEFAULT_SETTINGS)
            self._current_settings = merged

    def _valid_overrides(self, overrides: dict) -> dict:
        """Drops file values the API would reject, one key at a time, so one bad value costs only itself."""
        from api.validation import sanitize_settings

        clean = {}
        for key, value in overrides.items():
            items = value.items() if key == "camera" and isinstance(value, dict) else [(None, value)]
            for sub_key, sub_value in items:
                if sub_key is not None and sub_key.startswith("cached_pipeline_"):
                    # Written by CameraManager, which checks them against its own candidate list.
                    clean.setdefault("camera", {})[sub_key] = sub_value
                    continue
                candidate = {key: {sub_key: sub_value}} if sub_key is not None else {key: value}
                try:
                    accepted = sanitize_settings(candidate)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Ignoring invalid setting in {self.filepath}: {e}")
                    continue
                for k, v in accepted.items():
                    if isinstance(v, dict):
                        clean.setdefault(k, {}).update(v)
                    else:
                        clean[k] = v
        return clean

    def save(self) -> None:
        """Writes the overrides atomically so a crash can never leave a truncated file."""
        with self.lock:
            overrides = {
                k: v for k, v in self._current_settings.items()
                if k in DEFAULT_SETTINGS and v != DEFAULT_SETTINGS[k]
            }
            directory = os.path.dirname(os.path.abspath(self.filepath))
            fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".settings-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(overrides, f, indent=4)
                os.replace(tmp_path, self.filepath)
            except OSError as e:
                logger.error(f"Failed to save {self.filepath}: {e}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def get(self, key: str, default: Any = None) -> Any:
        with self.lock:
            return copy.deepcopy(self._current_settings.get(key, default))

    def __getattr__(self, name: str) -> Any:
        state = self.__dict__
        if "_current_settings" not in state or "lock" not in state:
            raise AttributeError(name)
        with state["lock"]:
            current = state["_current_settings"]
            if name == "slouch_ratio_threshold":
                return _SLOUCH_RATIO.get(current.get("slouch_sensitivity"), _SLOUCH_RATIO["Medium"])
            if name == "gaze_away_threshold":
                return _GAZE_AWAY_THRESHOLD.get(current.get("eye_tracking_sensitivity"), _GAZE_AWAY_THRESHOLD["Medium"])
            if name in current:
                return copy.deepcopy(current[name])
        raise AttributeError(f"'SettingsManager' object has no attribute '{name}'")

    def get_all(self) -> Dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self._current_settings)

    def update(self, new_settings: Dict[str, Any]) -> None:
        """Merges known keys into the current settings and persists them."""
        with self.lock:
            for k, v in new_settings.items():
                if k not in self._current_settings:
                    continue
                if isinstance(self._current_settings[k], dict) and isinstance(v, dict):
                    self._current_settings[k].update(copy.deepcopy(v))
                else:
                    self._current_settings[k] = copy.deepcopy(v)
            self.save()

    def reset_section(self, keys: list[str]) -> None:
        """Resets specific keys back to defaults."""
        with self.lock:
            for k in keys:
                if k in DEFAULT_SETTINGS:
                    self._current_settings[k] = copy.deepcopy(DEFAULT_SETTINGS[k])
            self.save()

    def restore_defaults(self) -> None:
        """Restores all settings to defaults."""
        with self.lock:
            self._current_settings = copy.deepcopy(DEFAULT_SETTINGS)
            self.save()


# Global Singleton
settings = SettingsManager()
