"""Regression tests for defects found in the 2026-09-30 review; each failed on commit 036cb54."""
import copy
import math
import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest

CSRF = {"X-DeskBot-Request": "1"}
W, H = 1280, 720


class _SilentAlerts:
    def __init__(self, *args, **kwargs):
        self.spoken = []

    def dispatch(self, text, category="general", cooldown=None, identity="Unknown"):
        self.spoken.append(text)


class _NoDetections:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, frame):
        return []

    def extract_pupil_gaze(self, frame, left_eye, right_eye, threshold=0.75):
        return 0.0, False


class _FakeProcess:
    def __init__(self, *args, **kwargs):
        self._alive = False

    def start(self):
        self._alive = True

    def is_alive(self):
        return self._alive

    def terminate(self):
        self._alive = False

    def kill(self):
        self._alive = False

    def join(self, timeout=None):
        pass


class _DummyEvaluator:
    def __init__(self):
        self.tracked_persons = {}
        self.profiles = {}
        self.active_sessions = {}
        self.primary_user_track_id = None
        self.pending_registration_name = None
        self.last_system_message = ""

    def sync_profiles(self):
        pass

    def request_registration(self, name):
        self.pending_registration_name = name

    def request_recalibration(self):
        pass

    def request_session_restart(self):
        pass


def _client(evaluator=None):
    from flask import Flask
    from api.routes import api_bp

    app = Flask(__name__)
    app.register_blueprint(api_bp)
    app.config["HEALTH_EVALUATOR"] = evaluator or _DummyEvaluator()
    return app.test_client()


def _unit(index, dim=128, noise=0.0, seed=0):
    vec = np.zeros(dim, dtype=np.float32)
    vec[index] = 1.0
    if noise:
        vec += np.random.default_rng(seed).normal(0, noise, dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _pose(l_sh, r_sh, nose, l_hip, r_hip):
    from core.tracking.geometry import Keypoint

    return {
        "nose": Keypoint(*nose, W, H),
        "left_shoulder": Keypoint(*l_sh, W, H),
        "right_shoulder": Keypoint(*r_sh, W, H),
        "left_hip": Keypoint(*l_hip, W, H),
        "right_hip": Keypoint(*r_hip, W, H),
        "left_ear": Keypoint(nose[0] - 40, nose[1], W, H),
        "right_ear": Keypoint(nose[0] + 40, nose[1], W, H),
    }


@pytest.fixture
def pristine_defaults():
    from config.defaults import DEFAULT_SETTINGS

    snapshot = copy.deepcopy(DEFAULT_SETTINGS)
    yield snapshot
    DEFAULT_SETTINGS.clear()
    DEFAULT_SETTINGS.update(snapshot)


def test_settings_update_keeps_defaults_and_persists_camera(tmp_path, pristine_defaults):
    from config.defaults import DEFAULT_SETTINGS
    from config.settings_manager import SettingsManager

    path = str(tmp_path / "settings.json")
    SettingsManager(filepath=path).update({"camera": {"preferred_camera": 2}})

    assert DEFAULT_SETTINGS == pristine_defaults
    assert SettingsManager(filepath=path).get("camera")["preferred_camera"] == 2


def test_health_eval_survives_collected_session():
    from core.tracking.orchestrator import TrackerEngine
    from core.tracking.state import Person

    engine = TrackerEngine.__new__(TrackerEngine)
    engine.active_sessions = {}
    engine.primary_user_track_id = "t1"
    engine.profiles = {}
    engine.alert_manager = _SilentAlerts()
    engine.last_system_message = ""

    person = Person("t1", None, [500.0, 100.0, 780.0, 700.0])
    person.name = "Alice"
    person.state = "Tracking Active"
    person.verification_status = "VERIFIED"
    person.posture_baseline = 1.0
    person.calibrated_baseline_neck_pitch = 5.0
    person.baseline_shoulder_y = 400 / H
    pose = _pose((560, 400), (720, 400), (640, 250), (580, 650), (700, 650))

    engine._evaluate_single_target_health(person, pose, 1.0, 0.1, time.monotonic(), (H, W, 3))

    assert "Alice" in engine.active_sessions
    assert engine.active_sessions["Alice"].posture_baseline == 1.0


def _evaluate(person_pitch, sensitivity, l_sh, r_sh, l_hip, r_hip):
    from posture.posture_analyzer import PostureAnalyzer

    nose = ((l_sh[0] + r_sh[0]) / 2, l_sh[1] - 150)
    pose = _pose(l_sh, r_sh, nose, l_hip, r_hip)
    shoulder_w = abs(pose["right_shoulder"].x - pose["left_shoulder"].x)
    ratio = abs((pose["left_shoulder"].y + pose["right_shoulder"].y) / 2 - pose["nose"].y) / shoulder_w
    person = SimpleNamespace(
        pitch=person_pitch, smoothed_pitch=person_pitch, slouch_sensitivity=sensitivity,
        box=[500.0, 100.0, 780.0, 700.0],
    )
    session = SimpleNamespace(
        calibrated_baseline_neck_pitch=5.0, posture_baseline=ratio,
        health_status="Healthy", baseline_shoulder_y=pose["left_shoulder"].y,
    )
    return PostureAnalyzer.evaluate(person, session, pose, (H, W, 3))


def test_posture_angles_are_measured_in_pixel_space():
    tilt = 200 * math.tan(math.radians(10))
    lean = 250 * math.tan(math.radians(20))
    hip_y = 400 + tilt / 2 + 250  # hip centre exactly 250 px below the shoulder centre
    # Facing the camera: the subject's left shoulder is on the image right (here also 10 degrees lower).
    state = _evaluate(5.0, 15.0, (740, 400 + tilt), (540, 400), (740 - lean, hip_y), (540 - lean, hip_y))

    assert abs(state.shoulder_alignment - 10.0) < 0.5
    assert abs(state.spine_alignment - 20.0) < 0.5


def test_per_user_slouch_sensitivity_is_used():
    state = _evaluate(25.0, 40.0, (560, 400), (720, 400), (580, 650), (700, 650))

    assert not state.is_slouching


def test_powershell_tts_keeps_text_out_of_the_script(monkeypatch):
    import alerts.alert_manager as am
    from utils.hardware import HardwareProfile

    calls = []

    class FakeProc:
        returncode = 0

        def __init__(self, args, **kwargs):
            calls.append((args, kwargs))

        def communicate(self, input=None, timeout=None):
            return b"", b""

    monkeypatch.setattr(am.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(am.shutil, "which", lambda name: "C:/tools/" + name)
    manager = am.AlertManager()
    manager.hardware = HardwareProfile("Windows", "AMD64", False, False, False, 8.0)
    payload = "D\u2019Souza'); Start-Process calc; ('"

    manager.dispatch(payload, category="t", cooldown=0.0, identity="x")
    manager.audio_queue.join()

    args, kwargs = calls[-1]
    assert "Start-Process" not in " ".join(args)
    assert (kwargs.get("env") or {}).get("DESKBOT_TTS_TEXT") == payload


def test_profile_create_rejects_markup_names():
    response = _client().post(
        "/api/profile/create", json={"name": "<img src=x onerror=alert(1)>"}, headers=CSRF
    )

    assert response.status_code == 400


def test_settings_api_rejects_capture_pipeline_injection(tmp_path, monkeypatch, pristine_defaults):
    from config import settings_manager as sm

    monkeypatch.setattr(sm.settings, "filepath", str(tmp_path / "s.json"))
    monkeypatch.setitem(sm.settings._current_settings, "camera", copy.deepcopy(pristine_defaults["camera"]))
    evil = {"init_arg": "http://attacker.example/x", "backend": 1900, "desc": "x"}

    _client().post("/api/settings", json={"camera": {"cached_pipeline_0": evil}}, headers=CSRF)

    assert "cached_pipeline_0" not in sm.settings.get("camera")


def test_state_changing_request_without_csrf_header_is_refused():
    assert _client().post("/api/profile/recalibrate").status_code == 403


def test_foreign_host_header_is_refused():
    assert _client().get("/api/profiles", headers={"Host": "evil.example"}).status_code == 400


def test_registration_survives_restart_and_delete_is_permanent(tmp_path, monkeypatch):
    import core.tracking.biometric_worker as worker
    import core.tracking.orchestrator as orch
    import database.crud as crud
    from core.tracking.state import Person

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DESKBOT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(orch, "PersonDetector", _NoDetections)
    monkeypatch.setattr(orch, "AlertManager", _SilentAlerts)
    monkeypatch.setattr(worker, "_mp", SimpleNamespace(Process=_FakeProcess, Queue=queue.Queue))

    db = crud.DatabaseManager()
    db.create_profile("Alice", _unit(0))
    engine = orch.TrackerEngine(db)
    engine.tracked_persons["t1"] = Person("t1", None, [540.0, 160.0, 740.0, 560.0])
    assert engine.request_registration("Bob") is None
    frame = np.zeros((H, W, 3), np.uint8)
    for i in range(5):
        engine.worker.result_queue.put({"track_id": "t1", "embedding": _unit(1, noise=0.01, seed=i)})
        engine.process_frame_mot(frame, frame.shape)

    restarted = orch.TrackerEngine(db)
    assert {"Alice", "Bob"} <= set(restarted.profiles)

    db.log_analytics_flush("Bob", "Tracking Active", 10.0, 1.0, 1.0, 0.0, 0.0)
    db.flush()
    assert restarted.remove_profile("Bob")
    db.flush()

    assert "Bob" not in orch.TrackerEngine(db).profiles
    assert db.get_user_analytics("Bob") == []
