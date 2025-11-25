"""
Pipeline-level latency tracking.

Tracks end-to-end pipeline latency for complete workflows.
"""

import time
from typing import Optional
from opentelemetry import trace


class PipelineTracker:
    """Tracks pipeline-level metrics including end-to-end latency."""
    
    def __init__(self, span: Optional[trace.Span] = None):
        """
        Initialize pipeline tracker.
        
        Args:
            span: Optional root span to attach metrics to
        """
        self.span = span
        self.start_time: Optional[float] = None
        
    def start(self) -> None:
        """Start tracking pipeline latency."""
        self.start_time = time.perf_counter()
        
    def stop(self) -> float:
        """
        Stop tracking and return latency in seconds.
        
        Returns:
            Latency in seconds
        """
        if self.start_time is None:
            raise ValueError("Pipeline tracker not started")
        
        latency_seconds = time.perf_counter() - self.start_time
        
        if self.span:
            latency_ms = latency_seconds * 1000
            self.span.set_attribute("pipeline.latency_ms", latency_ms)
        
        return latency_seconds
    
    def track_pipeline_latency(self, span: trace.Span, latency_seconds: float) -> None:
        """
        Track pipeline latency on a span.
        
        Args:
            span: Span to attach latency to
            latency_seconds: Latency in seconds
        """
        latency_ms = latency_seconds * 1000
        span.set_attribute("pipeline.latency_ms", latency_ms)

