"""Inference loop between the camera and the tracker, and the status the API reports about it."""
import logging
import threading
import time

import cv2

from camera.camera_manager import is_black
from config.settings_manager import settings

logger = logging.getLogger(__name__)

_JPEG_QUALITY = 80
_HIDDEN_STATES = ("Searching / Re-acquiring", "Absent")


class PipelineStatus:
    """Loop health plus the latest annotated JPEG, shared by the inference thread and HTTP handlers."""

    def __init__(self):
        self._cond = threading.Condition()
        self._jpeg = None
        self._sequence = 0
        self._viewers = 0
        self.frames_processed = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.last_error = ""
        self.last_frame_at = None
        self.black_since = None
        self.started_at = time.monotonic()

    @property
    def viewers(self) -> int:
        return self._viewers

    def add_viewer(self):
        with self._cond:
            self._viewers += 1

    def try_add_viewer(self, limit: int) -> bool:
        with self._cond:
            if self._viewers >= limit:
                return False
            self._viewers += 1
            return True

    def remove_viewer(self):
        with self._cond:
            self._viewers -= 1

    def publish(self, jpeg: bytes):
        with self._cond:
            self._jpeg = jpeg
            self._sequence += 1
            self._cond.notify_all()

    def wait_for_frame(self, last_sequence: int, timeout: float = 1.0):
        """Returns (sequence, jpeg); the sequence is unchanged if no new frame arrived in time."""
        with self._cond:
            self._cond.wait_for(lambda: self._sequence != last_sequence, timeout)
            return self._sequence, self._jpeg

    def record_success(self):
        self.frames_processed += 1
        self.consecutive_errors = 0
        self.last_frame_at = time.monotonic()

    def record_error(self, error: Exception):
        self.errors += 1
        self.consecutive_errors += 1
        self.last_error = f"{type(error).__name__}: {error}"

    def record_brightness(self, black: bool):
        if not black:
            self.black_since = None
        elif self.black_since is None:
            self.black_since = time.monotonic()

    def snapshot(self) -> dict:
        now = time.monotonic()
        age = None if self.last_frame_at is None else round(now - self.last_frame_at, 2)
        black_since = self.black_since
        return {
            "frames_processed": self.frames_processed,
            "errors": self.errors,
            "consecutive_errors": self.consecutive_errors,
            "last_error": self.last_error,
            "last_frame_age_s": age,
            "black_for_s": 0.0 if black_since is None else round(now - black_since, 1),
            "uptime_s": round(now - self.started_at, 1),
            "viewers": self._viewers,
        }


def annotate(frame, engine):
    """Draws tracked people onto the frame in place."""
    with engine.mutex:
        for person in engine.tracked_persons.values():
            if person.state in _HIDDEN_STATES:
                continue
            x1, y1, x2, y2 = (int(v) for v in person.box[:4])
            color = (0, 255, 135) if person.is_verified or person.name != "Unknown" else (67, 159, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{person.name} [{person.state}]", (x1, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )


def run_step(camera, engine, status: PipelineStatus) -> bool:
    """Processes one frame; returns False when the camera had no new frame."""
    frame = camera.get_frame()
    if frame is None:
        return False
    try:
        status.record_brightness(is_black(frame))
        engine.process_frame_mot(frame, frame.shape)
        if status.viewers > 0:
            annotate(frame, engine)
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            if ok:
                status.publish(jpeg.tobytes())
        status.record_success()
    except Exception as e:
        # One bad frame must not kill the loop; a persistent fault shows up in /api/system/health.
        status.record_error(e)
        if status.consecutive_errors == 1 or status.consecutive_errors % 100 == 0:
            logger.exception(f"Inference step failed ({status.consecutive_errors} in a row)")
    return True


def inference_loop(camera, engine, status: PipelineStatus, stop_event: threading.Event):
    while not stop_event.is_set():
        started = time.monotonic()
        if not run_step(camera, engine, status):
            stop_event.wait(0.01)
            continue
        min_interval = 1.0 / max(1, int(settings.get("detection_fps", 30)))
        remaining = min_interval - (time.monotonic() - started)
        if remaining > 0:
            stop_event.wait(remaining)
