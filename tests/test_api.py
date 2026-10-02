import io
import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest
from flask import Flask

from core.pipeline import PipelineStatus
from core.tracking.state import Person

CSRF = {"X-DeskBot-Request": "1"}


class FakeEngine:
    def __init__(self):
        self.tracked_persons = {}
        self.profiles = {"Alice": {}}
        self.active_sessions = {}
        self.primary_user_track_id = None
        self.last_system_message = ""
        self.person_detector = SimpleNamespace(session=None, providers=[])
        self.registrations = []

    def request_registration(self, name):
        self.registrations.append(name)

    def remove_profile(self, name):
        return self.profiles.pop(name, None) is not None

    def sync_profiles(self):
        pass

    def worker_status(self):
        return {"alive": False, "restarts": 3}


class FakeCameraBridge:
    def __init__(self):
        self.metrics = {"state": "reconnecting"}
        self.active_camera_info = None
        self.restarts = 0

    def discover_cameras(self, force=False):
        return [{"index": 0, "name": "Fake", "type": "USB", "path": "0", "status": "Available"}]

    def restart_async(self):
        self.restarts += 1


@pytest.fixture
def app(monkeypatch, tmp_path):
    from api import routes
    from config.settings_manager import SettingsManager

    monkeypatch.setattr(routes, "settings", SettingsManager(filepath=str(tmp_path / "settings.json")))
    flask_app = Flask(__name__)
    flask_app.register_blueprint(routes.api_bp)
    flask_app.config["HEALTH_EVALUATOR"] = FakeEngine()
    flask_app.config["CAMERA_BRIDGE"] = FakeCameraBridge()
    return flask_app


def test_health_names_every_broken_component(app):
    status = PipelineStatus()
    status.record_error(RuntimeError("detector exploded"))
    app.config["PIPELINE_STATUS"] = status

    data = app.test_client().get("/api/system/health").get_json()

    assert data["status"] == "degraded"
    reasons = " | ".join(data["reasons"])
    for expected in ("camera is reconnecting", "detector exploded", "pose model is not loaded", "worker is down"):
        assert expected in reasons


def test_health_reports_a_camera_stuck_on_black_frames(app, monkeypatch):
    import core.pipeline as pipeline

    clock = [1000.0]
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    status = PipelineStatus()
    status.record_success()
    app.config["PIPELINE_STATUS"] = status
    client = app.test_client()

    def black_reasons():
        return [r for r in client.get("/api/system/health").get_json()["reasons"] if "black" in r]

    status.black_since = clock[0] - 1.0
    assert black_reasons() == []
    status.black_since = clock[0] - 10.0
    reasons = black_reasons()
    assert len(reasons) == 1 and "black for 10s" in reasons[0] and "another app" in reasons[0]
    clock[0] += 30.0  # frames stopped arriving: the camera is gone, not black
    assert black_reasons() == []


def test_camera_select_validates_and_switches_in_background(app):
    client = app.test_client()
    bridge = app.config["CAMERA_BRIDGE"]
    assert client.post("/api/cameras/select", json={"index": "0"}, headers=CSRF).status_code == 400
    assert client.post("/api/cameras/select", json={"index": True}, headers=CSRF).status_code == 400
    assert client.post("/api/cameras/select", json={"index": 7}, headers=CSRF).status_code == 404
    assert bridge.restarts == 0

    assert client.post("/api/cameras/select", json={"index": 0}, headers=CSRF).status_code == 202
    assert bridge.restarts == 1


def test_cross_origin_post_is_refused_even_with_the_header(app):
    client = app.test_client()
    for origin in ("http://evil.example", "http://localhost:8080"):
        assert client.post("/api/profile/restart_session", headers={**CSRF, "Origin": origin}).status_code == 403
    same_origin = {**CSRF, "Origin": "http://localhost"}  # the test client's own origin
    assert client.post("/api/profile/create", json={"name": "Bob"}, headers=same_origin).status_code == 200


def test_cross_site_subresources_are_refused_but_links_still_work(app):
    client = app.test_client()
    probe = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"}
    assert client.get("/api/cameras", headers=probe).status_code == 403
    assert client.get("/api/profiles", headers={"Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "cors"}).status_code == 403
    link = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"}
    assert client.get("/api/profiles", headers=link).status_code == 200


def test_profile_endpoints_report_conflicts_and_missing_profiles(app):
    client = app.test_client()
    assert client.post("/api/profile/create", json={"name": "Alice"}, headers=CSRF).status_code == 409
    assert client.post("/api/profile/delete", json={"name": "Nobody"}, headers=CSRF).status_code == 404
    assert client.post("/api/profile/delete", json={"name": "Alice"}, headers=CSRF).status_code == 200
    assert client.post("/api/profile/create", data="not json", headers=CSRF).status_code == 400


def test_settings_import_rejects_bad_and_oversized_files(app):
    client = app.test_client()
    bad = {"file": (io.BytesIO(b"{not json"), "s.json")}
    assert client.post("/api/settings/import", data=bad, headers=CSRF, content_type="multipart/form-data").status_code == 400
    invalid = {"file": (io.BytesIO(b'{"detection_fps": "fast"}'), "s.json")}
    assert client.post("/api/settings/import", data=invalid, headers=CSRF, content_type="multipart/form-data").status_code == 400
    huge = {"file": (io.BytesIO(b" " * (2 * 1024 * 1024)), "s.json")}
    assert client.post("/api/settings/import", data=huge, headers=CSRF, content_type="multipart/form-data").status_code == 413


def test_invalid_settings_are_not_saved(app):
    from api import routes

    client = app.test_client()
    assert client.post("/api/settings", json={"detection_fps": 0}, headers=CSRF).status_code == 400
    assert routes.settings.get("detection_fps") == 30
    response = client.post("/api/settings", json={"detection_fps": 10}, headers=CSRF)
    assert response.status_code == 200 and routes.settings.get("detection_fps") == 10


def test_responses_forbid_framing_and_video_needs_a_pipeline(app):
    response = app.test_client().get("/video_feed")
    assert response.status_code == 503
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Cross-Origin-Resource-Policy"] == "same-origin"


def test_video_feed_refuses_cross_site_embedding_and_extra_viewers(app):
    status = PipelineStatus()
    app.config["PIPELINE_STATUS"] = status
    client = app.test_client()
    assert client.get("/video_feed", headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"}).status_code == 403

    response = client.get("/video_feed")
    assert response.status_code == 200 and status.viewers == 1
    response.close()  # what the server does when the client goes away
    assert status.viewers == 0

    for _ in range(3):
        status.add_viewer()
    assert client.get("/video_feed").status_code == 503


def test_video_stream_keeps_writing_without_new_frames():
    from api.routes import video_stream_generator

    stream = video_stream_generator(PipelineStatus())
    started = time.monotonic()
    part = next(stream)
    # Writing lets the server notice a closed client; a silent stream would pin a server thread.
    assert time.monotonic() - started < 2.0 and b"\xff\xd8" in part


def test_waitress_rejects_oversized_bodies_before_buffering_them():
    import http.client
    import threading

    from waitress import create_server

    import deskbot_v3

    server = create_server(deskbot_v3.app, host="127.0.0.1", port=0, **deskbot_v3.WAITRESS_OPTIONS)
    threading.Thread(target=server.run, daemon=True).start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", server.effective_port, timeout=10)
        conn.putrequest("POST", "/api/settings/import")
        conn.putheader("Content-Length", str(2 * 1024 * 1024))
        conn.putheader("X-DeskBot-Request", "1")
        conn.endheaders()  # the body is never sent: the limit must apply to the declared length
        assert conn.getresponse().status == 413
        conn.close()
    finally:
        server.close()


def test_health_flags_a_pipeline_without_frames_and_a_dead_db_writer(app):
    status = PipelineStatus()
    status.started_at -= 60
    app.config["PIPELINE_STATUS"] = status
    app.config["DB_CONN"] = SimpleNamespace(writer_alive=lambda: False)

    reasons = app.test_client().get("/api/system/health").get_json()["reasons"]

    assert "no frame processed since start" in reasons
    assert any("database writer" in r for r in reasons)


def test_legacy_profile_with_out_of_range_limit_stays_editable(app, tmp_path):
    from database.crud import DatabaseManager

    db = DatabaseManager(str(tmp_path / "profiles.db"))
    db.create_profile("Old", np.ones(128, np.float32), screen_gaze_limit=20)  # the old settings UI allowed 5-60 s
    app.config["DB_CONN"] = db
    client = app.test_client()

    assert client.post("/api/profile/update", json={"name": "Old", "session_limit": 1800}, headers=CSRF).status_code == 200
    profile = db.load_all_profiles()["Old"]
    assert (profile["session_limit"], profile["screen_gaze_limit"]) == (1800, 20)
    assert client.post("/api/profile/update", json={"name": "Old"}, headers=CSRF).status_code == 400
    assert client.post("/api/profile/update", json={"name": "Nobody", "session_limit": 1800}, headers=CSRF).status_code == 404


def _bare_engine():
    import core.tracking.orchestrator as orch
    from core.tracking.state import state_mutex

    engine = orch.TrackerEngine.__new__(orch.TrackerEngine)
    engine.mutex = state_mutex
    engine.tracked_persons, engine.profiles, engine.active_sessions = {}, {}, {}
    engine.pending_registration_name, engine.pending_registration_deadline = None, 0.0
    engine.primary_user_track_id = engine.current_authenticated_user = None
    engine.last_system_message = ""
    return engine


def test_registration_needs_exactly_one_unregistered_person_in_view():
    engine = _bare_engine()
    guest, other = Person("a", None, [0, 0, 10, 10]), Person("b", None, [20, 0, 30, 10])
    engine.tracked_persons = {"a": guest, "b": other}
    assert "More than one" in engine.request_registration("Carol")

    other.name, other.verification_status, other.state = "Alice", "VERIFIED", "Tracking Active"
    assert "registered user" in engine.request_registration("Carol")

    del engine.tracked_persons["b"]
    assert engine.request_registration("Carol") is None and engine.pending_registration_name == "Carol"


def test_pending_registration_expires_instead_of_enrolling_a_later_visitor():
    engine = _bare_engine()
    assert engine.request_registration("Carol") is None
    engine._expire_pending_registration(time.monotonic() + 61)
    assert engine.pending_registration_name is None
    assert "timed out" in engine.last_system_message


def test_face_matching_an_existing_profile_is_recognised_as_a_duplicate():
    from core.tracking.identity import matching_profile

    alice = np.zeros(128, np.float32)
    alice[0] = 1.0
    profiles = {"Alice": {"embeddings": alice[None, :]}}
    noisy = alice + np.random.default_rng(0).normal(0, 0.02, 128).astype(np.float32)
    stranger = np.roll(alice, 5)
    assert matching_profile(noisy, profiles, 0.25) == "Alice"
    assert matching_profile(stranger, profiles, 0.25) is None


def test_deleting_a_profile_forgets_the_live_identity():
    engine = _bare_engine()
    engine.db_manager = SimpleNamespace(delete_profile=lambda name: True)
    engine.profiles, engine.active_sessions = {"Alice": {}}, {"Alice": object()}
    person = Person("t", None, [0, 0, 10, 10])
    person.name, person.verification_status, person.state = "Alice", "VERIFIED", "Tracking Active"
    engine.tracked_persons, engine.primary_user_track_id, engine.current_authenticated_user = {"t": person}, "t", "Alice"

    assert engine.remove_profile("Alice")

    assert (person.name, person.verification_status) == ("Unknown", "UNKNOWN")
    assert engine.primary_user_track_id is None and engine.current_authenticated_user is None
    assert "Alice" not in engine.profiles and "Alice" not in engine.active_sessions


def _worker(monkeypatch, alive):
    import core.tracking.biometric_worker as bw

    terminated = []

    class Process:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return alive

        def terminate(self):
            terminated.append(True)

    monkeypatch.setattr(bw, "_mp", SimpleNamespace(Process=Process, Queue=queue.Queue))
    worker = bw.BiometricWorker()
    worker._started_at = 0.0
    return worker, terminated


def test_hung_worker_is_terminated_so_it_can_be_restarted(monkeypatch):
    worker, terminated = _worker(monkeypatch, alive=True)
    assert worker.submit("t", None)  # never taken by the worker
    worker.supervise(100.0)
    worker.supervise(120.0)
    assert terminated == []
    worker.supervise(131.0)
    assert terminated == [True]


def test_dead_face_worker_is_restarted_with_backoff(monkeypatch):
    worker, _ = _worker(monkeypatch, alive=False)
    started = []
    monkeypatch.setattr(worker, "start", lambda: started.append(True))

    assert worker.supervise(100.0)
    assert not worker.supervise(100.5)  # within the check interval
    assert not worker.supervise(101.5)  # within the 2 s backoff
    assert worker.supervise(102.5)
    assert len(started) == 2 and worker.restarts == 2

    worker.shutdown()
    assert not worker.supervise(200.0)


def test_engine_forgets_jobs_lost_with_a_restarted_worker(monkeypatch):
    engine = _bare_engine()
    engine.worker = SimpleNamespace(supervise=lambda now: True, results=lambda: iter(()))
    waiting = Person("t", None, [0, 0, 10, 10])
    waiting.verification_status = "VERIFYING"
    engine.tracked_persons = {"t": waiting}
    monkeypatch.setattr(engine, "_apply_manual_requests", lambda now: None)
    monkeypatch.setattr(engine, "_submit_biometric_job", lambda *args: None)
    engine.manual_recalibration_requested = engine.manual_session_restart_requested = False
    engine.anchor_lost_since, engine.last_session_owner = None, None

    engine._process_frame_locked(np.zeros((10, 10, 3), np.uint8), (10, 10, 3), [])

    assert waiting.verification_status == "UNKNOWN"


def test_live_metrics_report_calibration_state():
    from analytics.session_manager import SessionManager
    from core.tracking.state import Person, UserSession

    person = Person("t1", None, [0, 0, 10, 10])
    person.name, person.state, person.verification_status = "Alice", "Tracking Active", "VERIFIED"
    session = UserSession("Alice")
    session.is_posture_calibrated, session.calibrated_at = True, 1700000000.0
    session.posture_baseline, session.calibrated_baseline_neck_pitch = 0.5, 3.0
    engine = SimpleNamespace(
        profiles={"Alice": {}}, tracked_persons={"t1": person}, active_sessions={"Alice": session},
        primary_user_track_id="t1",
    )

    [row] = SessionManager.extract_live_metrics(engine)

    assert row["is_primary"]
    assert row["calibration"] == {
        "calibrated": True, "calibrated_at": 1700000000.0, "head": True, "shoulders": False, "back": True,
    }
