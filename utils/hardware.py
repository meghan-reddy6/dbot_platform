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


class HardwareDetector:
    """
    Detects hardware capabilities at runtime to automatically select the optimal
    inference backend and processing optimizations.
    """

    @staticmethod
    def _check_gpu() -> bool:
        # Simple heuristic, assumes if CUDA/ROCm or standard GPU libs are present
        return os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" or os.path.exists(
            "/dev/nvidia0"
        )

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
