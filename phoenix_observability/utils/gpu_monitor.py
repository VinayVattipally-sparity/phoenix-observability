"""
GPU monitoring utilities.

Optional GPU usage tracking for model servers.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import pynvml

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logger.warning(
        "pynvml not available. GPU monitoring disabled. "
        "Install with: pip install pynvml"
    )


class GPUMonitor:
    """Monitor GPU usage statistics."""

    def __init__(self):
        """Initialize GPU monitor."""
        self.initialized = False
        if GPU_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.initialized = True
                self.device_count = pynvml.nvmlDeviceGetCount()
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")
                self.initialized = False

    def get_gpu_stats(self, device_id: int = 0) -> Optional[Dict[str, any]]:
        """
        Get GPU statistics for a device.

        Args:
            device_id: GPU device ID

        Returns:
            Dictionary with GPU stats or None if unavailable
        """
        if not self.initialized:
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_mb = mem_info.used / (1024**2)
            memory_total_mb = mem_info.total / (1024**2)
            memory_utilization = (mem_info.used / mem_info.total) * 100

            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = util.gpu
            memory_utilization_pct = util.memory

            # Temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except:
                temperature = None

            # Power
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except:
                power = None

            return {
                "device_id": device_id,
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb,
                "memory_utilization_percent": memory_utilization,
                "gpu_utilization_percent": gpu_utilization,
                "memory_utilization_percent_nvml": memory_utilization_pct,
                "temperature_celsius": temperature,
                "power_watts": power,
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU stats for device {device_id}: {e}")
            return None

    def get_all_gpu_stats(self) -> Dict[int, Dict[str, any]]:
        """
        Get statistics for all GPUs.

        Returns:
            Dictionary mapping device ID to stats
        """
        if not self.initialized:
            return {}

        stats = {}
        for device_id in range(self.device_count):
            device_stats = self.get_gpu_stats(device_id)
            if device_stats:
                stats[device_id] = device_stats

        return stats

    def shutdown(self):
        """Shutdown GPU monitoring."""
        if self.initialized and GPU_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


# Global GPU monitor instance
_gpu_monitor: Optional[GPUMonitor] = None


def get_gpu_monitor() -> Optional[GPUMonitor]:
    """Get or create the global GPU monitor instance."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor

