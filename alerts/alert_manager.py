import threading
import time
import logging
import subprocess
import shutil
import queue
from utils.hardware import HardwareDetector

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages system alerts and TTS synthesis. Decoupled from state tracking
    to allow cross-platform compatibility (Windows PowerShell, Linux espeak,
    or completely silent for embedded NPU nodes without audio hardware).
    """

    def __init__(self, alert_cooldown: float = 20.0):
        self.alert_cooldown = alert_cooldown
        self.hardware = HardwareDetector.detect()
        
        # Identity-based cooldown tracking
        self.user_cooldowns = {}
        
        # Audio Pipeline
        self.audio_queue = queue.Queue()
        self.audio_thread_active = True
        self._worker_thread = threading.Thread(target=self._audio_worker_thread, daemon=True)
        self._worker_thread.start()

    def _audio_worker_thread(self):
        """Dedicated background thread to sequentially process audio alerts and wait for completion."""
        while self.audio_thread_active:
            try:
                text_prompt = self.audio_queue.get(timeout=1.0)
                
                # Disable TTS on embedded hardware without audio routing explicitly requested
                if self.hardware.is_embedded:
                    logger.debug(f"[SILENT ALERT]: {text_prompt}")
                    self.audio_queue.task_done()
                    continue

                logger.info(f"[AUDIO DEBUG] Speech Started: {text_prompt}")

                if shutil.which("powershell") and self.hardware.platform_system == "Windows":
                    ps_script = f"Add-Type -AssemblyName System.speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('{text_prompt.replace(chr(39), chr(39) + chr(39))}')"
                    proc = subprocess.Popen(
                        ["powershell", "-Command", ps_script],
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                    proc.communicate() # BLOCK UNTIL FINISHED
                elif shutil.which("espeak"):
                    proc = subprocess.Popen(["espeak", text_prompt])
                    proc.communicate()
                else:
                    logger.warning(f"[AUDIO ERROR] No TTS engine found on PATH. Alert: {text_prompt}")
                
                logger.info(f"[AUDIO DEBUG] Speech Completed: {text_prompt}")
                self.audio_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[AUDIO ERROR] Voice Daemon Exception: {e}")

    def dispatch(
        self, text: str, category: str = "general", cooldown: float = 30.0, identity: str = "Unknown"
    ) -> None:
        """
        Dispatches an audio alert ensuring identity-based cooldown limits are respected.
        """
        current_time = time.time()
        active_cooldown = cooldown if cooldown is not None else self.alert_cooldown
        
        if identity not in self.user_cooldowns:
            self.user_cooldowns[identity] = {}
            
        last_alert_time = self.user_cooldowns[identity].get(category, 0.0)
        
        is_cooldown_passed = (
            current_time - last_alert_time > active_cooldown
        ) or (category == "registration")

        if is_cooldown_passed:
            self.user_cooldowns[identity][category] = current_time
            logger.info(f"[AUDIO DEBUG] QUEUED | Category: {category} | Identity: {identity} | Text: {text}")
            self.audio_queue.put(text)
