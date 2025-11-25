"""
Latency measurement utilities.

Lightweight timer for measuring execution time.
"""

import time
from contextlib import contextmanager
from typing import Optional


class LatencyTimer:
    """Simple timer for measuring latency."""

    def __init__(self):
        """Initialize the timer."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()

    def stop(self) -> float:
        """
        Stop the timer and return elapsed time in seconds.

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise ValueError("Timer not started")
        self.end_time = time.perf_counter()
        return self.elapsed()

    def elapsed(self) -> float:
        """
        Get elapsed time without stopping.

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.perf_counter()
        return end - self.start_time

    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = None
        self.end_time = None

    @contextmanager
    def measure(self):
        """
        Context manager for measuring latency.

        Usage:
            with timer.measure() as elapsed:
                # do work
            print(f"Took {elapsed()} seconds")
        """
        self.start()
        try:
            yield self
        finally:
            self.stop()


def measure_latency(func):
    """
    Decorator to measure function execution latency.

    Args:
        func: Function to measure

    Returns:
        Wrapped function that measures latency
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timer = LatencyTimer()
        timer.start()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = timer.stop()
            # Attach to function if it has a latency attribute
            if hasattr(func, "__latency__"):
                func.__latency__ = elapsed
    return wrapper


def track_pipeline_latency(span, latency_seconds: float) -> None:
    """
    Track pipeline-level latency in milliseconds.

    Args:
        span: OpenTelemetry span
        latency_seconds: Latency in seconds
    """
    latency_ms = latency_seconds * 1000
    span.set_attribute("pipeline.latency_ms", latency_ms)

