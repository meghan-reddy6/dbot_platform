import json
import os
import threading
from typing import Dict, Any
from config.defaults import DEFAULT_SETTINGS

class SettingsManager:
    """
    Manages loading, saving, and merging configuration overrides.
    Uses data/settings.json to persist user overrides.
    """
    
    def __init__(self, filepath="data/settings.json"):
        self.filepath = filepath
        self.lock = threading.Lock()
        self._current_settings = DEFAULT_SETTINGS.copy()
        
        # Ensure data dir exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self.load()
        
    def load(self) -> None:
        """Loads overrides from settings.json and merges with defaults."""
        with self.lock:
            if not os.path.exists(self.filepath):
                self._current_settings = DEFAULT_SETTINGS.copy()
                return
                
            try:
                with open(self.filepath, 'r') as f:
                    overrides = json.load(f)
                    
                # Merge overrides into defaults
                merged = DEFAULT_SETTINGS.copy()
                for k, v in overrides.items():
                    if k in merged:
                        merged[k] = v
                        
                self._current_settings = merged
            except Exception as e:
                print(f"[SettingsManager] Failed to load {self.filepath}: {e}")
                self._current_settings = DEFAULT_SETTINGS.copy()

    def save(self) -> None:
        """Extracts only overrides and saves to settings.json."""
        with self.lock:
            overrides = {}
            for k, v in self._current_settings.items():
                if k in DEFAULT_SETTINGS and v != DEFAULT_SETTINGS[k]:
                    overrides[k] = v
                    
            try:
                with open(self.filepath, 'w') as f:
                    json.dump(overrides, f, indent=4)
            except Exception as e:
                print(f"[SettingsManager] Failed to save {self.filepath}: {e}")
                
    def get(self, key: str, default: Any = None) -> Any:
        with self.lock:
            return self._current_settings.get(key, default)
            
    def __getattr__(self, name: str) -> Any:
        with self.lock:
            # Dynamic attribute mapping for legacy/computed fields
            if name == "slouch_ratio_threshold":
                sens = self._current_settings.get("slouch_sensitivity", "Medium")
                if sens == "Low": return 0.70
                if sens == "High": return 0.90
                return 0.80
            
            if name in self._current_settings:
                return self._current_settings[name]
            raise AttributeError(f"'SettingsManager' object has no attribute '{name}'")
            
    def get_all(self) -> Dict[str, Any]:
        with self.lock:
            return self._current_settings.copy()
            
    def update(self, new_settings: Dict[str, Any]) -> None:
        """Updates settings in memory and triggers a save."""
        with self.lock:
            for k, v in new_settings.items():
                if k in self._current_settings:
                    self._current_settings[k] = v
        self.save()
        
    def reset_section(self, keys: list[str]) -> None:
        """Resets specific keys back to defaults."""
        with self.lock:
            for k in keys:
                if k in DEFAULT_SETTINGS:
                    self._current_settings[k] = DEFAULT_SETTINGS[k]
        self.save()
        
    def restore_defaults(self) -> None:
        """Restores all settings to defaults."""
        with self.lock:
            self._current_settings = DEFAULT_SETTINGS.copy()
        self.save()

# Global Singleton
settings = SettingsManager()
