import os
import platform
import psutil
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Preference order; filtered at runtime to what the installed onnxruntime build provides.
_PREFERRED_PROVIDERS = [
    "QNNExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]
_QNN_EP = "QNNExecutionProvider"
_qnn_plugin_devices = None
# ONNX Runtime's fused int8 CPU kernels on 64-bit ARM compute the quantized face detector wrongly (1.27 and 1.30
# on a Snapdragon QCS8550: outputs off by up to 125 quantization steps, so no face is ever found). Unfused
# (DequantizeLinear -> float op -> QuantizeLinear) it finds the same face as on x86. Float models have no QDQ
# nodes to fuse. session.qdqisint8allowed=0 and mlas.disable_kleidiai=1 do not help.
_ARM64 = platform.machine().lower() in ("aarch64", "arm64")
_arm64_note_logged = False


def qnn_plugin_devices() -> list:
    """NPU devices of the onnxruntime-qnn >= 2.0 plugin; loads and registers the plugin on first call.

    Set DESKBOT_DISABLE_QNN=1 to skip it, e.g. if the native plugin crashes or hangs on a board.
    """
    global _qnn_plugin_devices
    if _qnn_plugin_devices is None:
        _qnn_plugin_devices = []
        if os.environ.get("DESKBOT_DISABLE_QNN") == "1":
            return _qnn_plugin_devices
        try:
            import onnxruntime as ort
            import onnxruntime_qnn
        except ImportError:
            return _qnn_plugin_devices
        try:
            ort.register_execution_provider_library(_QNN_EP, onnxruntime_qnn.get_library_path())
            # Only a real NPU counts: on x86 hosts the plugin lists its HTP emulator on a CPU device.
            _qnn_plugin_devices = [
                d for d in ort.get_ep_devices()
                if d.ep_name == _QNN_EP and d.device.type == ort.OrtHardwareDeviceType.NPU
            ]
        except Exception as e:
            logger.warning(f"onnxruntime-qnn is installed but its execution provider could not be registered: {e}")
    return _qnn_plugin_devices


def available_ort_providers() -> set:
    try:
        import onnxruntime as ort
    except ImportError:
        return set()
    providers = set(ort.get_available_providers())
    if qnn_plugin_devices():
        providers.add(_QNN_EP)
    return providers


def create_session(path: str, providers: list):
    """InferenceSession for a provider list; when it starts with the QNN plugin, the session runs on the NPU
    (HTP backend) with ONNX Runtime's CPU EP taking any unsupported nodes."""
    import onnxruntime as ort

    global _arm64_note_logged
    devices = qnn_plugin_devices()
    if devices and providers and providers[0] == _QNN_EP:
        import onnxruntime_qnn

        # No disable_quant_qdq here: it lets ONNX Runtime constant-fold DequantizeLinear, so the NPU would get
        # float weights instead of the quantized model. The NPU ran every node of the face detector.
        options = ort.SessionOptions()
        options.add_provider_for_devices(devices, {"backend_path": onnxruntime_qnn.get_qnn_htp_path()})
        return ort.InferenceSession(path, sess_options=options)
    if devices:
        providers = [p for p in providers if p != _QNN_EP] or ["CPUExecutionProvider"]
    if _ARM64 and providers == ["CPUExecutionProvider"]:
        if not _arm64_note_logged:
            _arm64_note_logged = True
            logger.info("64-bit ARM CPU: quantized models run without int8 fusion (ONNX Runtime's fused kernels "
                        "give wrong face detections here).")
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_quant_qdq", "1")
        return ort.InferenceSession(path, sess_options=options, providers=providers)
    return ort.InferenceSession(path, providers=providers)


@dataclass
class HardwareProfile:
    platform_system: str
    platform_machine: str
    is_embedded: bool
    has_gpu: bool
    has_npu: bool
    total_ram_gb: float

    def get_ort_providers(self) -> list:
        available = available_ort_providers()
        return [p for p in _PREFERRED_PROVIDERS if p in available] or ["CPUExecutionProvider"]


class HardwareDetector:
    """
    Detects hardware capabilities at runtime to automatically select the optimal
    inference backend and processing optimizations.
    """

    @staticmethod
    def _check_gpu() -> bool:
        return bool(available_ort_providers() & {"CUDAExecutionProvider", "DmlExecutionProvider"})

    @staticmethod
    def _check_npu() -> bool:
        # True with a QNN-enabled onnxruntime build (onnxruntime-qnn 1.x) or the onnxruntime-qnn >= 2.0 plugin.
        return "QNNExecutionProvider" in available_ort_providers()

    @staticmethod
    def _is_embedded(system: str, machine: str) -> bool:
        # Apple Silicon and Windows-on-ARM report arm64 too, but they are desktops.
        return system == "Linux" and machine.lower() in ("aarch64", "armv7l", "arm64")

    _cached_profile: HardwareProfile = None

    @classmethod
    def detect(cls) -> HardwareProfile:
        if cls._cached_profile is not None:
            return cls._cached_profile

        system = platform.system()
        machine = platform.machine()

        is_embedded = cls._is_embedded(system, machine)

        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024**3)

        has_gpu = cls._check_gpu()
        has_npu = cls._check_npu()

        profile = HardwareProfile(
            platform_system=system,
            platform_machine=machine,
            is_embedded=is_embedded,
            has_gpu=has_gpu,
            has_npu=has_npu,
            total_ram_gb=round(ram_gb, 2),
        )

        logger.info(f"Hardware Detected: {profile}")
        cls._cached_profile = profile
        return profile
