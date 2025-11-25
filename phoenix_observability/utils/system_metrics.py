"""
System metrics tracking utilities.

Tracks CPU, memory, and GPU usage.
"""

import logging
import psutil
from typing import Dict, Optional

from phoenix_observability.utils.gpu_monitor import get_gpu_monitor

logger = logging.getLogger(__name__)


def get_system_metrics() -> Dict[str, any]:
    """
    Get current system metrics.
    
    Returns:
        Dictionary with system metrics
    """
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        memory_percent = memory.percent
        
        # GPU metrics
        gpu_monitor = get_gpu_monitor()
        gpu_available = gpu_monitor is not None and gpu_monitor.initialized
        gpu_stats = None
        if gpu_available:
            gpu_stats = gpu_monitor.get_gpu_stats(0)
        
        return {
            "cpu_percent": cpu_percent,
            "memory_used_gb": round(memory_used_gb, 2),
            "memory_available_gb": round(memory_available_gb, 2),
            "memory_percent": round(memory_percent, 1),
            "gpu": {
                "available": gpu_available,
                "stats": gpu_stats
            }
        }
    except Exception as e:
        logger.warning(f"Failed to get system metrics: {e}")
        return {}


def attach_system_metrics_to_span(span, include_gpu: bool = True) -> None:
    """
    Attach system metrics to a span.
    
    Args:
        span: OpenTelemetry span
        include_gpu: Whether to include GPU metrics
    """
    try:
        metrics = get_system_metrics()
        
        # CPU
        if "cpu_percent" in metrics:
            span.set_attribute("system.cpu_percent", metrics["cpu_percent"])
        
        # Memory
        if "memory_used_gb" in metrics:
            span.set_attribute("system.memory_used_gb", metrics["memory_used_gb"])
        if "memory_available_gb" in metrics:
            span.set_attribute("system.memory_available_gb", metrics["memory_available_gb"])
        if "memory_percent" in metrics:
            span.set_attribute("system.memory_percent", metrics["memory_percent"])
        
        # GPU
        if include_gpu and "gpu" in metrics:
            gpu_info = metrics["gpu"]
            span.set_attribute("system.gpu.available", gpu_info["available"])
            if gpu_info.get("stats"):
                gpu_stats = gpu_info["stats"]
                if "gpu_utilization_percent" in gpu_stats:
                    span.set_attribute("system.gpu.utilization_percent", gpu_stats["gpu_utilization_percent"])
                if "memory_used_mb" in gpu_stats:
                    span.set_attribute("system.gpu.memory_used_mb", gpu_stats["memory_used_mb"])
    except Exception as e:
        logger.debug(f"Failed to attach system metrics: {e}")

