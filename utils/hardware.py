import platform
import os
import psutil
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    platform_system: str
    platform_machine: str
    is_embedded: bool
    has_gpu: bool
    has_npu: bool
    total_ram_gb: float

    def get_ort_providers(self) -> list:
        providers = []
        if self.has_npu:
            providers.append('QNNExecutionProvider')
            providers.append('NnapiExecutionProvider')
        if self.has_gpu:
            providers.append('CUDAExecutionProvider')
            providers.append('DmlExecutionProvider') # DirectML for Windows GPUs
        providers.append('CPUExecutionProvider')
        return providers


class HardwareDetector:
    """
    Detects hardware capabilities at runtime to automatically select the optimal
    inference backend and processing optimizations.
    """

    @staticmethod
    def _check_gpu() -> bool:
        # Check for CUDA (NVIDIA)
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or os.path.exists("/dev/nvidia0"):
            return True
        # Check for AMD/Intel GPUs on Windows via ONNX DirectML provider availability
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'DmlExecutionProvider' in providers:
                return True
        except ImportError:
            pass
            
        # Fallback to checking registry or OS specific indicators if needed
        if platform.system() == "Windows":
            return True # Windows almost always has a GPU (Intel/AMD/Nvidia) capable of DirectML
            
        return False

    @staticmethod
    def _check_npu() -> bool:
        # Check for Rubik Pi / QCS6490 NPU nodes
        return os.path.exists("/dev/qce") or os.path.exists("/dev/kgsl-3d0")

    _cached_profile: HardwareProfile = None

    @classmethod
    def detect(cls) -> HardwareProfile:
        if cls._cached_profile is not None:
            return cls._cached_profile

        system = platform.system()
        machine = platform.machine()

        # Check if running on Rubik Pi or similar ARM embedded board
        is_embedded = machine.lower() in ["aarch64", "armv7l", "arm64"]

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
