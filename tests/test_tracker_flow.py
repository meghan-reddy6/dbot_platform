"""Characterization test for TrackerEngine's frame loop: pins the observable state trace so refactors stay behaviour-neutral."""
import itertools
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from core.tracking.geometry import Keypoint

W, H = 1280, 720
FPS = 30.0


class _Clock:
    def __init__(self, start):
        self.now = start

    def monotonic(self):
        return self.now

    def time(self):
        return 1_700_000_000.0 + self.now


class _Alerts:
    def __init__(self, *args, **kwargs):
        self.spoken = []

    def dispatch(self, text, category="general", cooldown=None, identity="Unknown"):
        self.spoken.append(category)


class _Process:
    def __init__(self, *args, **kwargs):
        self.alive = False

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    def join(self, timeout=None):
        pass


def _seated_person(dx=0.0, slouch=False):
    """One upright person at desk distance (over 15 % of the frame), facing the camera."""
    nose_y = 330 if slouch else 230  # head dropped towards the shoulders
    points = {
        "nose": (640 + dx, nose_y), "left_ear": (700 + dx, 215), "right_ear": (580 + dx, 215),
        "left_shoulder": (840 + dx, 420), "right_shoulder": (440 + dx, 420),
        "left_hip": (800 + dx, 700), "right_hip": (480 + dx, 700),
    }
    pose = {k: Keypoint(x, y, W, H) for k, (x, y) in points.items()}
    return {
        "box": np.array([420.0 + dx, 150.0, 860.0 + dx, 710.0]),
        "pose": pose,
        "nose": np.array(points["nose"]),
        "l_eye": np.array([675.0 + dx, 200.0]),
        "r_eye": np.array([605.0 + dx, 200.0]),
        "pitch": 2.0, "yaw": 0.0, "roll": 0.0,
    }


class _Script:
    """Detections per frame: one seated person for `present` frames (slouching in `slouch`), then nobody."""

    def __init__(self, present, slouch=range(0), absent=range(0)):
        self.present = present
        self.slouch = slouch
        self.absent = absent
        self.frame = 0

    def detect(self, frame):
        self.frame += 1
        if self.frame in self.absent:
            return []
        return [_seated_person(slouch=self.frame in self.slouch)] if self.frame <= self.present else []

    def extract_pupil_gaze(self, frame, left_eye, right_eye, threshold=0.75):
        return 0.0, False


def _unit(i):
    v = np.zeros(128, np.float32)
    v[i] = 1.0
    return v


@pytest.fixture
def make_engine(monkeypatch, tmp_path):
    import core.tracking.biometric_worker as worker
    import core.tracking.health as health
    import core.tracking.orchestrator as orch
    import core.tracking.state as state
    from database.crud import DatabaseManager

    def make(script, **limits):
        # A fixed start: whether a 1/30 s step lands just before or after a threshold depends on the clock's value.
        clock = _Clock(start=1000.0)
        fake_time = SimpleNamespace(monotonic=clock.monotonic, time=clock.time)
        for module in (orch, state, health, worker):
            monkeypatch.setattr(module, "time", fake_time)
        monkeypatch.setattr(worker, "_mp", SimpleNamespace(Process=_Process, Queue=queue.Queue))
        db = DatabaseManager(str(tmp_path / "flow.db"))
        db.create_profile("Alice", _unit(0), **limits)
        tracker = orch.TrackerEngine(db, person_detector=script, alert_manager=_Alerts())
        tracker.clock = clock
        return tracker

    return make


def _run(engine, frames):
    """Steps the loop at 30 FPS, answering face jobs as the worker would, and returns a compressed trace."""
    frame = np.zeros((H, W, 3), np.uint8)
    trace = []
    for _ in range(frames):
        engine.clock.now += 1.0 / FPS
        engine.process_frame_mot(frame, frame.shape)
        while not engine.worker.job_queue.empty():
            job = engine.worker.job_queue.get_nowait()
            engine.worker.result_queue.put({"track_id": job["track_id"], "embedding": _unit(0)})
        people = list(engine.tracked_persons.values())
        trace.append(
            tuple((p.name, str(p.state), str(p.verification_status), p.health_status) for p in people)
            + (("primary", engine.primary_user_track_id is not None),)
        )
    return [(state, len(list(group))) for state, group in itertools.groupby(trace)]


def test_frame_loop_trace_is_unchanged(make_engine):
    engine = make_engine(_Script(present=150))
    trace = _run(engine, 330)

    assert trace == EXPECTED_TRACE
    assert engine.alert_manager.spoken == ["calibration_prompt", "calibration_success"]


def test_slouch_trace_is_unchanged(make_engine):
    engine = make_engine(_Script(present=420, slouch=range(150, 330)))
    trace = _run(engine, 420)

    assert trace == EXPECTED_SLOUCH_TRACE
    assert engine.alert_manager.spoken == ["calibration_prompt", "calibration_success", "posture_alert"]


def test_screen_time_and_sitting_limit_trace_is_unchanged(make_engine):
    engine = make_engine(_Script(present=900), session_limit=20, screen_gaze_limit=10, ocular_break_duration=2)
    trace = _run(engine, 900)

    assert trace == EXPECTED_TIMER_TRACE
    assert engine.alert_manager.spoken == ["calibration_prompt", "calibration_success", "ocular_break", "session_limit"]


def test_a_video_call_pauses_tracking_instead_of_ending_the_session(make_engine):
    # Frames 201-215 come back without a detection, as while a camera adjusts after another app released it.
    engine = make_engine(_Script(present=10_000, absent=range(201, 216)))
    _run(engine, 200)
    session = engine.active_sessions["Alice"]
    tracks, sitting = set(engine.tracked_persons), session.sitting_duration_clock
    for _ in range(600):  # ten minutes of black frames: the pipeline holds the tracks
        engine.clock.now += 1.0
        engine.hold_tracks()
    _run(engine, 45)

    assert set(engine.tracked_persons) == tracks
    assert engine.active_sessions["Alice"] is session
    assert session.sitting_duration_clock - sitting < 3.5  # 1 s of frames plus one capped step, not 600 s


# Recorded from the pre-refactor engine. 150 frames present (5 s), then gone: recognised after 5 worker
# replies, calibrated over 3 s, tracked, held 2 s after leaving, searched 3 s, then evicted.
EXPECTED_TRACE = [
    ((("Unknown", "Unregistered Guest", "VERIFYING", "Healthy"), ("primary", False)), 5),
    ((("Alice", "Calibrating", "VERIFIED", "Healthy"), ("primary", True)), 92),
    ((("Alice", "Tracking Active", "VERIFIED", "Healthy"), ("primary", True)), 113),
    ((("Alice", "Searching / Re-acquiring", "VERIFIED", "Healthy"), ("primary", False)), 90),
    ((("primary", False),), 30),
]
# Slouching from frame 150 to 330: the alert fires after 4 s and clears 1.5 s after sitting up.
EXPECTED_SLOUCH_TRACE = [
    ((("Unknown", "Unregistered Guest", "VERIFYING", "Healthy"), ("primary", False)), 5),
    ((("Alice", "Calibrating", "VERIFIED", "Healthy"), ("primary", True)), 92),
    ((("Alice", "Tracking Active", "VERIFIED", "Healthy"), ("primary", True)), 172),
    ((("Alice", "Tracking Active", "VERIFIED", "Posture Deficit Alert"), ("primary", True)), 105),
    ((("Alice", "Tracking Active", "VERIFIED", "Healthy"), ("primary", True)), 46),
]
# Limits of 10 s screen time and 20 s sitting: the eye-break reminder, then the stand-up reminder.
EXPECTED_TIMER_TRACE = [
    ((("Unknown", "Unregistered Guest", "VERIFYING", "Healthy"), ("primary", False)), 5),
    ((("Alice", "Calibrating", "VERIFIED", "Healthy"), ("primary", True)), 92),
    ((("Alice", "Tracking Active", "VERIFIED", "Healthy"), ("primary", True)), 301),
    ((("Alice", "Tracking Active", "VERIFIED", "Ocular Break Recommended"), ("primary", True)), 300),
    ((("Alice", "Tracking Active", "VERIFIED", "Session Limit Reached - Stand Up!"), ("primary", True)), 202),
]
