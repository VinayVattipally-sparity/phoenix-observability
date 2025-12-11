"""
Metrics collection utilities.

Provides metrics for key operations like span creation, cost calculation, etc.
"""

import time
import threading
from typing import Dict, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricCounter:
    """Counter metric for tracking counts."""
    name: str
    count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def increment(self, value: int = 1) -> None:
        """Increment counter by value."""
        with self._lock:
            self.count += value
    
    def get(self) -> int:
        """Get current count."""
        with self._lock:
            return self.count
    
    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self.count = 0


@dataclass
class MetricHistogram:
    """Histogram metric for tracking distributions."""
    name: str
    values: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def record(self, value: float) -> None:
        """Record a value."""
        with self._lock:
            self.values.append(value)
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistics (min, max, avg, count)."""
        with self._lock:
            if not self.values:
                return {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0}
            
            values = self.values.copy()
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }
    
    def reset(self) -> None:
        """Reset histogram."""
        with self._lock:
            self.values.clear()


class MetricsCollector:
    """
    Central metrics collector for observability operations.
    
    Thread-safe metrics collection for:
    - Span creation counts
    - Cost calculation operations
    - Latency measurements
    - Error counts
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self._counters: Dict[str, MetricCounter] = {}
        self._histograms: Dict[str, MetricHistogram] = {}
        self._lock = threading.Lock()
    
    def get_counter(self, name: str) -> MetricCounter:
        """Get or create a counter metric."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = MetricCounter(name)
            return self._counters[name]
    
    def get_histogram(self, name: str) -> MetricHistogram:
        """Get or create a histogram metric."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = MetricHistogram(name)
            return self._histograms[name]
    
    def increment_span_created(self, span_type: str = "default") -> None:
        """Increment span creation counter."""
        counter = self.get_counter(f"spans.created.{span_type}")
        counter.increment()
        # Also increment total
        self.get_counter("spans.created.total").increment()
    
    def record_span_latency(self, latency_ms: float, span_type: str = "default") -> None:
        """Record span latency."""
        histogram = self.get_histogram(f"spans.latency.{span_type}")
        histogram.record(latency_ms)
        # Also record in total
        self.get_histogram("spans.latency.total").record(latency_ms)
    
    def increment_cost_calculated(self, model: Optional[str] = None) -> None:
        """Increment cost calculation counter."""
        if model:
            counter = self.get_counter(f"cost.calculated.{model}")
            counter.increment()
        self.get_counter("cost.calculated.total").increment()
    
    def record_cost_amount(self, cost: float, model: Optional[str] = None) -> None:
        """Record cost amount."""
        if model:
            histogram = self.get_histogram(f"cost.amount.{model}")
            histogram.record(cost)
        self.get_histogram("cost.amount.total").record(cost)
    
    def increment_error(self, error_type: str = "unknown") -> None:
        """Increment error counter."""
        counter = self.get_counter(f"errors.{error_type}")
        counter.increment()
        self.get_counter("errors.total").increment()
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary."""
        with self._lock:
            metrics: Dict[str, Any] = {
                "counters": {},
                "histograms": {},
            }
            
            for name, counter in self._counters.items():
                metrics["counters"][name] = counter.get()
            
            for name, histogram in self._histograms.items():
                metrics["histograms"][name] = histogram.get_stats()
            
            return metrics
    
    def reset_all(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector (thread-safe)."""
    global _metrics_collector
    if _metrics_collector is None:
        with _metrics_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector()
    return _metrics_collector


def reset_metrics() -> None:
    """Reset all metrics (useful for testing)."""
    collector = get_metrics_collector()
    collector.reset_all()


# Convenience functions for common metrics
def record_span_created(span_type: str = "default") -> None:
    """Record a span creation."""
    get_metrics_collector().increment_span_created(span_type)


def record_span_latency(latency_ms: float, span_type: str = "default") -> None:
    """Record span latency."""
    get_metrics_collector().record_span_latency(latency_ms, span_type)


def record_cost_calculated(model: Optional[str] = None) -> None:
    """Record a cost calculation."""
    get_metrics_collector().increment_cost_calculated(model)


def record_cost_amount(cost: float, model: Optional[str] = None) -> None:
    """Record cost amount."""
    get_metrics_collector().record_cost_amount(cost, model)


def record_error(error_type: str = "unknown") -> None:
    """Record an error."""
    get_metrics_collector().increment_error(error_type)

