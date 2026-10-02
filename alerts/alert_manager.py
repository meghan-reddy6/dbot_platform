import threading
import time
import logging
import os
import subprocess
import shutil
import queue
from config.settings_manager import settings
from utils.hardware import HardwareDetector

logger = logging.getLogger(__name__)

# The text travels in an environment variable so PowerShell never parses it as code
# (it also treats U+2018-U+201B as quote characters, which defeats escaping).
_POWERSHELL_TTS_SCRIPT = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.Speak($env:DESKBOT_TTS_TEXT)"
)
_TTS_TIMEOUT_SECONDS = 60
REMINDER_CATEGORIES = {"ocular_break", "session_limit", "posture_alert"}


def _windows_powershell():
    # A bare "powershell" is resolved against the working directory first on Windows.
    system_copy = os.path.join(
        os.environ.get("SystemRoot", r"C:\Windows"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
    )
    return system_copy if os.path.isfile(system_copy) else shutil.which("powershell")


def build_tts_command(text, platform_system):
    """Returns (argv, extra_env, stdin_bytes) for the first available TTS engine, or None."""
    if platform_system == "Windows":
        powershell = _windows_powershell()
        if powershell:
            argv = [powershell, "-NoProfile", "-NonInteractive", "-Command", _POWERSHELL_TTS_SCRIPT]
            return argv, {"DESKBOT_TTS_TEXT": text}, None
    if platform_system == "Darwin" and shutil.which("say"):
        return [shutil.which("say"), "-f", "-"], None, text.encode("utf-8")
    for engine in ("espeak-ng", "espeak"):
        path = shutil.which(engine)
        if path:
            return [path, "--stdin"], None, text.encode("utf-8")
    return None


class AlertManager:
    """
    Manages system alerts and TTS synthesis on a dedicated thread so speech never blocks inference.
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
            except queue.Empty:
                continue
            try:
                self._speak(text_prompt)
            except Exception as e:
                logger.error(f"[AUDIO ERROR] Voice Daemon Exception: {e}")
            finally:
                self.audio_queue.task_done()

    def _speak(self, text_prompt):
        if not settings.get("enable_sound_alerts", True):
            logger.debug(f"[SILENT ALERT] Sound alerts disabled: {text_prompt}")
            return
        command = build_tts_command(text_prompt, self.hardware.platform_system)
        if command is None:
            logger.warning(f"[AUDIO ERROR] No TTS engine found on PATH. Alert: {text_prompt}")
            return

        argv, extra_env, stdin_bytes = command
        kwargs = {}
        if extra_env:
            kwargs["env"] = {**os.environ, **extra_env}
        if stdin_bytes is not None:
            kwargs["stdin"] = subprocess.PIPE
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        logger.info(f"[AUDIO DEBUG] Speech Started: {text_prompt}")
        proc = subprocess.Popen(argv, **kwargs)
        try:
            proc.communicate(input=stdin_bytes, timeout=_TTS_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        logger.info(f"[AUDIO DEBUG] Speech Completed: {text_prompt}")

    def dispatch(
        self, text: str, category: str = "general", cooldown: float = 30.0, identity: str = "Unknown"
    ) -> None:
        """
        Dispatches an audio alert ensuring identity-based cooldown limits are respected.
        """
        current_time = time.monotonic()
        active_cooldown = cooldown if cooldown is not None else self.alert_cooldown
        if category in REMINDER_CATEGORIES:
            active_cooldown = max(active_cooldown, settings.get("reminder_cooldown", 0))

        last_alert_time = self.user_cooldowns.setdefault(identity, {}).get(category)
        is_cooldown_passed = (
            last_alert_time is None
            or current_time - last_alert_time >= active_cooldown
            or category == "registration"
        )

        if is_cooldown_passed:
            self.user_cooldowns[identity][category] = current_time
            logger.info(f"[AUDIO DEBUG] QUEUED | Category: {category} | Identity: {identity} | Text: {text}")
            self.audio_queue.put(text)
