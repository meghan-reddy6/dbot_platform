import pytest

from alerts import alert_manager as am
from api.validation import (
    attachment_header,
    sanitize_settings,
    validate_profile_name,
    validate_profile_update,
    validate_settings_keys,
)
from utils.hardware import HardwareDetector


@pytest.mark.parametrize("name", ["Alice", "Jos\u00e9 \u00c1lvarez", "D\u2019Souza", "O'Neil", "Anne-Marie", "\u674e\u96f7", "R2 D2"])
def test_real_names_are_accepted(name):
    assert validate_profile_name(name) == name


@pytest.mark.parametrize(
    "name", ["", "   ", "<b>x</b>", 'a"b', "Unknown", "unknown guest", "-dash", "x" * 41, None, 5, "a/b", "tab\tname"]
)
def test_markup_reserved_and_malformed_names_are_rejected(name):
    with pytest.raises(ValueError):
        validate_profile_name(name)


def test_profile_update_keeps_legacy_names_editable_but_checks_limits():
    assert validate_profile_update(
        {"name": "old_name!", "slouch_sensitivity": 20, "session_limit": 1200, "stand_requirement": 120,
         "screen_gaze_limit": 1200, "ocular_break_duration": 900}
    )["name"] == "old_name!"
    with pytest.raises(ValueError):
        validate_profile_update({"name": "Alice", "slouch_sensitivity": 20, "session_limit": 5})


def test_settings_are_typed_and_unknown_keys_dropped():
    clean = sanitize_settings({
        "detection_fps": 15.0,
        "enable_sound_alerts": False,
        "confidence_threshold": 80,
        "camera": {"fps": 30, "cached_pipeline_0": {"init_arg": "evil ! appsink"}},
    })
    assert clean == {"detection_fps": 15, "enable_sound_alerts": False, "camera": {"fps": 30}}


@pytest.mark.parametrize(
    "payload",
    [
        {"detection_fps": True},
        {"detection_fps": "30"},
        {"detection_fps": 0},
        {"detection_fps": float("nan")},
        {"detection_fps": 10 ** 400},
        {"detection_fps": 12.5},
        {"slouch_sensitivity": "Extreme"},
        {"slouch_sensitivity": ["High"]},
        {"camera": {"resolution": [10, 10]}},
        {"camera": "x"},
        ["not", "an", "object"],
    ],
)
def test_invalid_settings_are_rejected(payload):
    with pytest.raises(ValueError):
        sanitize_settings(payload)


def test_reset_keys_are_limited_to_known_settings():
    assert validate_settings_keys(["detection_fps", "__class__"]) == ["detection_fps"]
    with pytest.raises(ValueError):
        validate_settings_keys("detection_fps")


def test_attachment_header_cannot_inject_headers():
    value = attachment_header('evil"\r\nSet-Cookie: x=1_analytics.csv')
    assert "\r" not in value and "\n" not in value
    assert value.count('"') == 2


def test_tts_text_never_reaches_a_command_line(monkeypatch):
    monkeypatch.setattr(am.shutil, "which", lambda name: "/usr/bin/" + name)
    text = "'; rm -rf ~ # \u2019); Start-Process calc"
    for system in ("Windows", "Darwin", "Linux"):
        argv, env, stdin = am.build_tts_command(text, system)
        assert text not in " ".join(argv)
        assert (env or {}).get("DESKBOT_TTS_TEXT") == text or stdin == text.encode("utf-8")


def test_missing_tts_engine_is_reported_not_crashed(monkeypatch):
    monkeypatch.setattr(am.shutil, "which", lambda name: None)
    assert am.build_tts_command("hello", "Linux") is None


@pytest.mark.parametrize(
    "system,machine,expected",
    [("Linux", "aarch64", True), ("Linux", "x86_64", False), ("Darwin", "arm64", False), ("Windows", "ARM64", False)],
)
def test_only_linux_arm_boards_count_as_embedded(system, machine, expected):
    assert HardwareDetector._is_embedded(system, machine) is expected


class _FakeSessionOptions:
    def __init__(self):
        self.added = []

    def add_provider_for_devices(self, devices, options):
        self.added.append((devices, options))


def _fake_qnn_plugin(monkeypatch, register_error=None, device_types=("GPU", "NPU")):
    import sys
    import types

    import onnxruntime as ort

    from utils import hardware

    qnn_devices = [types.SimpleNamespace(ep_name="QNNExecutionProvider",
                                         device=types.SimpleNamespace(type=getattr(ort.OrtHardwareDeviceType, t)))
                   for t in device_types]
    plugin = types.SimpleNamespace(get_library_path=lambda: "/opt/libonnxruntime_providers_qnn.so",
                                   get_qnn_htp_path=lambda: "/opt/libQnnHtp.so")
    monkeypatch.setitem(sys.modules, "onnxruntime_qnn", plugin)
    monkeypatch.setattr(hardware, "_qnn_plugin_devices", None)
    monkeypatch.delenv("DESKBOT_DISABLE_QNN", raising=False)
    registered = []

    def register(name, path):
        registered.append((name, path))
        if register_error:
            raise RuntimeError(register_error)

    monkeypatch.setattr(ort, "register_execution_provider_library", register)
    cpu = types.SimpleNamespace(ep_name="CPUExecutionProvider",
                                device=types.SimpleNamespace(type=ort.OrtHardwareDeviceType.CPU))
    monkeypatch.setattr(ort, "get_ep_devices", lambda: [cpu, *qnn_devices])
    monkeypatch.setattr(ort, "SessionOptions", _FakeSessionOptions)
    created = []
    monkeypatch.setattr(ort, "InferenceSession", lambda path, **kw: created.append(kw) or kw)
    return hardware, qnn_devices, created, registered


def test_qnn_plugin_ep_is_registered_once_and_runs_sessions_on_the_npu(monkeypatch):
    hardware, (_, npu), created, registered = _fake_qnn_plugin(monkeypatch)

    assert "QNNExecutionProvider" in hardware.available_ort_providers()
    hardware.create_session("m.onnx", ["QNNExecutionProvider", "CPUExecutionProvider"])
    # A providers list next to sess_options would override the device options and silently disable the NPU.
    assert set(created[-1]) == {"sess_options"}
    assert created[-1]["sess_options"].added == [([npu], {"backend_path": "/opt/libQnnHtp.so"})]
    hardware.create_session("m.onnx", ["CPUExecutionProvider"])
    assert created[-1] == {"providers": ["CPUExecutionProvider"]}
    hardware.available_ort_providers()
    assert registered == [("QNNExecutionProvider", "/opt/libonnxruntime_providers_qnn.so")]


def test_qnn_plugin_without_an_npu_device_is_not_used(monkeypatch):
    # x86 hosts list the plugin's HTP emulator on a CPU device; it must not outrank CUDA or DirectML.
    hardware, _, _, _ = _fake_qnn_plugin(monkeypatch, device_types=("CPU", "GPU"))
    assert "QNNExecutionProvider" not in hardware.available_ort_providers()


def test_a_qnn_plugin_that_fails_to_load_leaves_the_other_providers_working(monkeypatch):
    hardware, _, created, _ = _fake_qnn_plugin(monkeypatch, register_error="libcdsprpc.so: cannot open")

    assert "QNNExecutionProvider" not in hardware.available_ort_providers()
    hardware.create_session("m.onnx", ["CPUExecutionProvider"])
    assert created[-1] == {"providers": ["CPUExecutionProvider"]}


def test_qnn_plugin_can_be_switched_off(monkeypatch):
    hardware, _, _, registered = _fake_qnn_plugin(monkeypatch)
    monkeypatch.setenv("DESKBOT_DISABLE_QNN", "1")
    assert "QNNExecutionProvider" not in hardware.available_ort_providers()
    assert registered == []


def test_legacy_qnn_onnxruntime_build_keeps_its_providers_list(monkeypatch):
    import sys

    hardware, _, created, _ = _fake_qnn_plugin(monkeypatch)
    monkeypatch.setitem(sys.modules, "onnxruntime_qnn", None)  # onnxruntime-qnn 1.x has no plugin module
    hardware.create_session("m.onnx", ["QNNExecutionProvider", "CPUExecutionProvider"])
    assert created[-1] == {"providers": ["QNNExecutionProvider", "CPUExecutionProvider"]}


@pytest.mark.parametrize("arm64", [True, False])
def test_quantized_models_run_unfused_on_the_cpu_of_arm64_boards(monkeypatch, arm64):
    import onnxruntime as ort

    from utils import hardware

    monkeypatch.setattr(hardware, "_qnn_plugin_devices", [])
    monkeypatch.setattr(hardware, "_ARM64", arm64)
    created = []
    monkeypatch.setattr(ort, "InferenceSession", lambda path, **kw: created.append(kw) or kw)

    hardware.create_session("face_detector.onnx", ["CPUExecutionProvider"])
    options = created[-1].get("sess_options")
    assert created[-1]["providers"] == ["CPUExecutionProvider"]
    if arm64:
        assert options.get_session_config_entry("session.disable_quant_qdq") == "1"
    else:
        assert options is None
    hardware.create_session("face_detector.onnx", ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert "sess_options" not in created[-1]  # the workaround is verified for CPU-only sessions only


def test_the_arm64_session_setting_really_changes_how_the_face_detector_runs(monkeypatch):
    # ONNX Runtime ignores unknown config keys, so a misspelt key would silently keep the faulty fused kernels.
    # On x86 the fused and unfused results differ slightly; identical outputs mean the key had no effect.
    from pathlib import Path

    import numpy as np

    from utils import hardware

    model = str(Path(__file__).resolve().parents[1] / "models" / "face_detector.onnx")
    image = np.random.default_rng(0).integers(0, 256, (1, 3, 256, 256), dtype=np.uint8)
    monkeypatch.setattr(hardware, "_qnn_plugin_devices", [])
    outputs = {}
    for arm64 in (False, True):
        monkeypatch.setattr(hardware, "_ARM64", arm64)
        outputs[arm64] = hardware.create_session(model, ["CPUExecutionProvider"]).run(None, {"image": image})
    assert any(not np.array_equal(a, b) for a, b in zip(outputs[False], outputs[True]))


def test_face_model_cpu_fallback_goes_through_the_session_factory(monkeypatch):
    from recognition import face_recognizer
    from utils import hardware

    calls = []

    def factory(path, providers):
        calls.append(providers)
        if providers != ["CPUExecutionProvider"]:
            raise RuntimeError("QNN backend failed to initialise")
        return "cpu session"

    monkeypatch.setattr(hardware, "create_session", factory)
    session = face_recognizer._create_session("face_detector.onnx", ["QNNExecutionProvider", "CPUExecutionProvider"])
    assert session == "cpu session"
    assert calls == [["QNNExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]]
