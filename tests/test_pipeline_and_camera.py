import threading
import time

import numpy as np
import pytest

import camera.camera_manager as cm
from config.settings_manager import SettingsManager
from core.pipeline import PipelineStatus, inference_loop, run_step


def _frame():
    return np.zeros((48, 64, 3), np.uint8)


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)

    def get_frame(self):
        return self.frames.pop(0) if self.frames else None


class FakeEngine:
    def __init__(self, fail_times=0):
        self.mutex = threading.RLock()
        self.tracked_persons = {}
        self.fail_times = fail_times
        self.calls = 0

    def process_frame_mot(self, frame, frame_shape):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("detector exploded")


def test_a_failing_frame_does_not_stop_the_loop():
    camera, engine, status = FakeCamera([_frame(), _frame(), _frame()]), FakeEngine(fail_times=1), PipelineStatus()
    for _ in range(3):
        assert run_step(camera, engine, status)
    assert (status.errors, status.frames_processed, status.consecutive_errors) == (1, 2, 0)
    assert "detector exploded" in status.last_error
    assert not run_step(camera, engine, status)


def test_black_frames_are_timed_until_a_real_frame_arrives():
    status, engine = PipelineStatus(), FakeEngine()
    dim = np.full((48, 64, 3), 8, np.uint8)  # a dark room is dim, not all zero
    run_step(FakeCamera([dim]), engine, status)
    assert status.black_since is None

    run_step(FakeCamera([_frame()]), engine, status)
    status.black_since -= 5.0  # the black frames started 5 s ago
    run_step(FakeCamera([_frame()]), engine, status)
    assert status.snapshot()["black_for_s"] >= 5.0

    run_step(FakeCamera([dim]), engine, status)
    assert status.snapshot()["black_for_s"] == 0.0
    assert not cm.is_black(np.zeros((0, 0, 3), np.uint8))  # an empty frame must not crash the capture thread
    assert cm.is_black(np.full((48, 64), 4, np.uint8)) and not cm.is_black(np.full((48, 64), 5, np.uint8))


def test_jpeg_is_encoded_only_while_someone_watches():
    status = PipelineStatus()
    run_step(FakeCamera([_frame()]), FakeEngine(), status)
    assert status.wait_for_frame(0, timeout=0.01) == (0, None)

    status.add_viewer()
    run_step(FakeCamera([_frame()]), FakeEngine(), status)
    sequence, jpeg = status.wait_for_frame(0, timeout=0.01)
    assert sequence == 1 and jpeg[:2] == b"\xff\xd8"


def test_inference_loop_stops_on_request():
    stop = threading.Event()
    thread = threading.Thread(target=inference_loop, args=(FakeCamera([]), FakeEngine(), PipelineStatus(), stop))
    thread.start()
    stop.set()
    thread.join(timeout=2)
    assert not thread.is_alive()


class FakeCapture:
    instances = []
    fail_reads_after = None
    level = 120

    def __init__(self, *args):
        self.released = False
        self.reads = 0
        self.level = FakeCapture.level  # like DirectShow: a black stream stays black until it is reopened
        FakeCapture.instances.append(self)

    def isOpened(self):
        return not self.released

    def read(self):
        time.sleep(0.002)
        self.reads += 1
        if FakeCapture.fail_reads_after is not None and self.reads > FakeCapture.fail_reads_after:
            return False, None
        return True, np.full((4, 4, 3), self.level, np.uint8)

    def set(self, *args):
        return True

    def get(self, prop):
        return 4

    def release(self):
        self.released = True


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def manager(monkeypatch, tmp_path):
    FakeCapture.instances = []
    FakeCapture.fail_reads_after = None
    FakeCapture.level = 120
    monkeypatch.setattr(cm.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cm, "settings", SettingsManager(filepath=str(tmp_path / "settings.json")))
    camera = cm.CameraManager()
    camera_info = {"index": 0, "name": "Fake", "type": "USB", "path": "0", "status": "Available"}
    monkeypatch.setattr(camera, "discover_cameras", lambda force=False: [camera_info])
    init_arg, backend, _ = camera._candidate_pipelines(0)[0]
    monkeypatch.setattr(
        camera, "_find_best_pipeline",
        lambda idx, stop_event=None: {"init_arg": init_arg, "backend": backend, "desc": "Fake"},
    )
    yield camera
    camera.stop()


def _capture_threads():
    return [t for t in threading.enumerate() if t.name == "camera-capture" and t.is_alive()]


def test_restart_leaves_one_capture_thread_and_releases_every_handle(manager):
    manager.start()
    manager.start()  # a second start must not spawn a second reader
    assert _wait(lambda: manager.get_frame() is not None)
    assert len(_capture_threads()) == 1

    manager.restart()
    assert _wait(lambda: manager.get_frame() is not None)
    assert len(_capture_threads()) == 1

    manager.stop()
    assert _capture_threads() == []
    assert manager.metrics["state"] == "stopped"
    assert all(cap.released for cap in FakeCapture.instances)
    assert manager.get_frame() is None


def test_read_failure_reconnects_and_caches_the_pipeline(manager):
    FakeCapture.fail_reads_after = 3
    manager.start()
    assert _wait(lambda: manager.metrics["reconnects"] >= 1)
    FakeCapture.fail_reads_after = None
    assert _wait(lambda: manager.metrics["state"] == "streaming" and manager.get_frame() is not None)
    assert cm.settings.get("camera")["cached_pipeline_0"]["desc"] == "Fake"


def test_slow_black_frames_make_the_camera_reopen_until_the_picture_returns(manager, monkeypatch):
    monkeypatch.setattr(cm, "_BLACK_REOPEN_SECONDS", 0.05)
    monkeypatch.setattr(cm, "_BUSY_MAX_FPS", 1e6)  # the fake reads fast; count its black frames as the busy stream
    info = manager.discover_cameras()[0]
    forced = []
    monkeypatch.setattr(manager, "discover_cameras", lambda force=False: forced.append(force) or [info])

    def lit_frame():
        frame = manager.get_frame()
        return frame is not None and frame.max() > 0

    monkeypatch.setattr(FakeCapture, "level", 0)  # another app is streaming from the camera
    manager.start()
    assert _wait(lambda: manager.metrics["reconnects"] >= 1)
    FakeCapture.level = 120  # and has let go of it; only a reopened capture sees the picture
    assert _wait(lit_frame)
    assert len(FakeCapture.instances) >= 2 and FakeCapture.instances[0].released
    assert True not in forced  # the same device is reopened; other cameras are not probed


def test_black_frames_at_the_normal_rate_are_not_reopened(manager, monkeypatch):
    monkeypatch.setattr(cm, "_BLACK_REOPEN_SECONDS", 0.05)
    monkeypatch.setattr(FakeCapture, "level", 0)  # a dark room: black, but streaming at full rate
    manager.start()
    assert _wait(lambda: manager.get_frame() is not None)
    time.sleep(0.5)  # about ten reopen windows
    assert manager.metrics["reconnects"] == 0 and len(FakeCapture.instances) == 1


def test_planted_capture_pipeline_is_never_opened(manager):
    planted = {"init_arg": "http://attacker.example/x", "backend": 1900, "desc": "x"}
    cm.settings.update({"camera": {"cached_pipeline_0": planted}})
    opened = []
    real_open = manager._open_pipeline
    manager._open_pipeline = lambda pipeline: opened.append(pipeline) or real_open(pipeline)

    manager.start()
    assert _wait(lambda: manager.get_frame() is not None)
    assert opened and planted not in opened


def test_stop_interrupts_a_running_benchmark(monkeypatch, tmp_path):
    FakeCapture.instances = []
    FakeCapture.fail_reads_after = None
    monkeypatch.setattr(cm.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(cm, "settings", SettingsManager(filepath=str(tmp_path / "settings.json")))
    camera = cm.CameraManager()
    info = {"index": 0, "name": "Fake", "type": "USB", "path": "0", "status": "Available"}
    monkeypatch.setattr(camera, "discover_cameras", lambda force=False: [info])
    monkeypatch.setattr(camera, "_candidate_pipelines", lambda idx: [(idx, 0, "A"), (idx, 0, "B"), (idx, 0, "C")])

    camera.start()
    assert _wait(lambda: len(FakeCapture.instances) >= 1)  # benchmarking: each candidate would take 2 s
    started = time.monotonic()
    camera.stop()

    assert time.monotonic() - started < 1.0
    assert _capture_threads() == []
    assert all(cap.released for cap in FakeCapture.instances)


def test_linux_discovery_puts_capture_nodes_first_and_drops_codec_and_metadata_nodes(monkeypatch):
    import io

    # The Qualcomm RB5 Gen 2 lists /dev/video1, 0, 33, 32 in this order: CamX nodes held by the camera
    # server (cannot be opened) and the video codec. A USB webcam adds a capture node and a metadata node.
    can_capture = {"/dev/video32": False, "/dev/video33": False, "/dev/video2": True, "/dev/video3": False}
    names = {0: "cam-req-mgr", 1: "cam_sync", 2: "HP 5MP Camera", 3: "HP 5MP Camera", 32: "", 33: ""}
    probed = []

    def sysfs_open(path, mode="r"):
        return io.StringIO(names[int(path.split("/")[-2][5:])] + "\n")

    monkeypatch.setattr(cm.glob, "glob", lambda pattern: ["/dev/video1", "/dev/video0", "/dev/video33",
                                                          "/dev/video32", "/dev/video3", "/dev/video2"])
    monkeypatch.setattr(cm, "open", sysfs_open, raising=False)
    monkeypatch.setattr(cm, "v4l2_can_capture", lambda dev: probed.append(dev) or can_capture[dev])
    camera = cm.CameraManager()
    camera.os_type = "Linux"
    assert [c["index"] for c in camera._probe_cameras()] == [2, 0, 1]
    assert "/dev/video0" not in probed and "/dev/video1" not in probed  # the camera server's nodes


@pytest.mark.parametrize("caps, device_caps, expected", [
    (0x84A04000, 0x04A04000, False),  # msm_vidc decoder on the RB5 Gen 2 (memory-to-memory codec)
    (0x84208003, 0x04208003, False),  # a codec that also sets the capture and output bits
    (0x84A00001, 0x04200001, True),   # UVC webcam capture node
    (0x84A00001, 0x04A00000, False),  # its metadata node: only the device-wide caps say capture
    (0x84201000, 0x04201000, True),   # multi-planar capture
    (0xA4201000, 0x24201000, None),   # media-controller capture node (needs media-ctl/libcamera first)
    (0x00000001, 0x00000000, True),   # old driver without per-node caps
])
def test_v4l2_capabilities_decide_what_is_a_camera(monkeypatch, caps, device_caps, expected):
    import struct
    import sys
    import types

    opened, closed = [], []

    def ioctl(fd, request, buf):
        assert fd == 3 and request == 0x80685600 and len(buf) == 104
        out = bytearray(buf)
        struct.pack_into("II", out, 84, caps, device_caps)
        return bytes(out)

    monkeypatch.setitem(sys.modules, "fcntl", types.SimpleNamespace(ioctl=ioctl))
    monkeypatch.setattr(cm, "os", types.SimpleNamespace(open=lambda *a: opened.append(a) or 3,
                                                        close=closed.append, O_RDWR=2, O_NONBLOCK=2048))
    assert cm.v4l2_can_capture("/dev/video9") is expected
    assert opened == [("/dev/video9", 2 | 2048)] and closed == [3]


def test_v4l2_node_whose_query_fails_is_unknown_and_closed(monkeypatch):
    import errno
    import sys
    import types

    closed = []

    def ioctl(fd, request, buf):
        raise OSError(errno.ENOTTY, "Inappropriate ioctl for device")

    monkeypatch.setitem(sys.modules, "fcntl", types.SimpleNamespace(ioctl=ioctl))
    monkeypatch.setattr(cm, "os", types.SimpleNamespace(open=lambda *a: 3, close=closed.append,
                                                        O_RDWR=2, O_NONBLOCK=2048))
    assert cm.v4l2_can_capture("/dev/video9") is None
    assert closed == [3]


def test_v4l2_node_that_cannot_be_opened_is_unknown(monkeypatch):
    import errno
    import sys
    import types

    def busy(*args):
        raise OSError(errno.EALREADY, "Operation already in progress")

    monkeypatch.setitem(sys.modules, "fcntl", types.SimpleNamespace(ioctl=None))
    monkeypatch.setattr(cm, "os", types.SimpleNamespace(open=busy, close=None, O_RDWR=2, O_NONBLOCK=2048))
    assert cm.v4l2_can_capture("/dev/video0") is None
