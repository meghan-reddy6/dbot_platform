import threading
import time
import logging
import subprocess
import shutil
from utils.hardware import HardwareDetector

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manages system alerts and TTS synthesis. Decoupled from state tracking
    to allow cross-platform compatibility (Windows PowerShell, Linux espeak,
    or completely silent for embedded NPU nodes without audio hardware).
    """

    def __init__(self, alert_cooldown: float = 20.0):
        self.last_alert_time = 0.0
        self.alert_cooldown = alert_cooldown
        self.audio_thread_active = False
        self.hardware = HardwareDetector.detect()

    def _say_via_subprocess(self, text_prompt: str):
        def worker():
            try:
                # Disable TTS on embedded hardware without audio routing explicitly requested
                if self.hardware.is_embedded:
                    logger.debug(f"[SILENT ALERT]: {text_prompt}")
                    return

                if (
                    shutil.which("powershell")
                    and self.hardware.platform_system == "Windows"
                ):
                    ps_script = f"Add-Type -AssemblyName System.speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('{text_prompt.replace(chr(39), chr(39) + chr(39))}')"
                    subprocess.Popen(
                        ["powershell", "-Command", ps_script],
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                elif shutil.which("espeak"):
                    subprocess.Popen(["espeak", text_prompt])
                else:
                    logger.warning(f"No TTS engine found on PATH. Alert: {text_prompt}")
            except Exception as e:
                logger.error(f"Voice Daemon Error: {e}")
            finally:
                self.audio_thread_active = False

        threading.Thread(target=worker, daemon=True).start()

    def dispatch(
        self, text: str, category: str = "general", cooldown: float = 30.0
    ) -> None:
        """
        Dispatches an audio alert ensuring cooldown limits are respected.
        """
        current_time = time.time()
        is_cooldown_passed = (
            current_time - self.last_alert_time > self.alert_cooldown
        ) or (category == "registration")

        if is_cooldown_passed and not self.audio_thread_active:
            self.last_alert_time = current_time
            self.audio_thread_active = True
            logger.info(f"[VOICE ALERT DISPATCHED]: {text}")
            self._say_via_subprocess(text)
