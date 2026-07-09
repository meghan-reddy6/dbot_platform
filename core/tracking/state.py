import threading
import time
import numpy as np
from collections import deque

state_mutex = threading.Lock()

class VoiceAlertDaemon:
    def __init__(self):
        self.last_alert_time = 0.0
        self.alert_cooldown = 20.0
        self.audio_thread_active = False

    def _say_via_subprocess(self, text_prompt: str):
        def worker():
            try:
                import subprocess
                import shutil
                if shutil.which("powershell"):
                    ps_script = f"Add-Type -AssemblyName System.speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('{text_prompt.replace(chr(39), chr(39)+chr(39))}')"
                    subprocess.Popen(["powershell", "-Command", ps_script], creationflags=subprocess.CREATE_NO_WINDOW)
                elif shutil.which("espeak"):
                    subprocess.Popen(["espeak", text_prompt])
                else:
                    print("[VOICE DAEMON ERROR] No TTS engine found on PATH (powershell/espeak).")
            except Exception as e:
                print(f"[VOICE DAEMON ERROR] {e}")
            finally:
                self.audio_thread_active = False
                
        threading.Thread(target=worker, daemon=True).start()

    def dispatch(self, text: str, category: str = "general", cooldown: float = 30.0) -> None:
        current_time = time.time()
        is_cooldown_passed = (current_time - self.last_alert_time > self.alert_cooldown) or (category == "registration")
        if is_cooldown_passed and not self.audio_thread_active:
            self.last_alert_time = current_time
            self.audio_thread_active = True
            self._say_via_subprocess(text)

class Person:
    """
    Represents a tracked human target containing bounding box logic, biometric identity,
    and temporal state logic for posture evaluation.
    """
    def __init__(self, track_id: int, embedding: np.ndarray, box: list) -> None:
        self.track_id = track_id
        if embedding is None:
            self.embedding = np.zeros(128, dtype=np.float32)
        else:
            self.embedding = embedding
        self.box = box
        self.name = "Unknown"
        self.state = "Unregistered Guest"
        self.last_state = "Unregistered Guest"
        self.verification_status = "UNKNOWN"
        self.biometric_match_counter = 0
        self.candidate_name = None
        self.verified_name = None
        self.frame_val_name = None
        self.verification_timer = 0.0
        self.lost_grace_timer = 0.0
        self.is_verified = False
        self.is_posture_calibrated = False
        
        self.last_seen = time.time()
        self.last_update = time.time()
        
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.gaze_x = 0.0
        
        self.is_standing = False
        self.is_looking_away = False
        
        self.sitting_duration_clock = 0.0
        self.standing_duration_clock = 0.0
        
        self.screen_gaze_accumulation_timer = 0.0
        self.ocular_break_timer = 0.0
        self.gaze_away_clock = 0.0
        self.slouch_timer = 0.0
        
        self.sustained_slouch_debounce_timer = 0.0
        self.tracking_active_debounce_timer = 0.0
        
        self.ocular_break_announced = False
        self.session_limit_announced = False
        self.slouch_announced = False
        
        self.slouch_sensitivity = 15.0
        self.session_limit = 2400
        self.stand_requirement = 120
        self.gaze_away_limit = 20.0
        self.screen_gaze_limit = 1200.0
        self.biometric_cutoff = 0.55
        self.last_analytics_flush_time = 0.0
        
        self.baseline_torso_ratio = 0.0     
        self.calibrated_baseline_neck_pitch = 0.0           
        self.baseline_shoulder_y = 0.0      
        self.calibration_accumulator = []  
        self.calibration_start = None
        self.biometric_consensus_frame_counter = 0
        self.calibration_announced = False  
        self.last_log_time = 0.0            
        
        self.state_history_window = deque(maxlen=25)  
        self.recovery_calibration_start = None
        self.recovery_accumulator = []
        
        self.smoothed_pitch = None
        self.smoothed_ratio = None
        self.smoothed_y = None
        self.last_y = 0.0

    def get_centroid(self):
        return ((self.box[0] + self.box[2])/2, (self.box[1] + self.box[3])/2)
