"""
Pipeline instrumentation wrapper.

Tracks end-to-end pipeline latency for complete workflows (RAG, agents, etc.).
"""

import functools
import logging
from typing import Any, Callable, Optional

from opentelemetry import trace

from phoenix_observability.otel_setup import get_tracer
from phoenix_observability.utils.latency import LatencyTimer

logger = logging.getLogger(__name__)


def instrument_pipeline(
    pipeline_name: Optional[str] = None,
):
    """
    Decorator to instrument complete pipelines (e.g., RAG pipelines, agent workflows).
    
    Tracks pipeline.latency_ms for end-to-end execution time.
    
    Args:
        pipeline_name: Name of the pipeline (defaults to function name)
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = pipeline_name or func.__name__

            # Create root span for the pipeline
            with tracer.start_as_current_span(f"pipeline.{name}") as span:
                # Set OpenInference span kind for pipelines
                span.set_attribute("openinference.span.kind", "CHAIN")
                
                span.set_attribute("pipeline.name", name)
                span.set_attribute("pipeline.function", func.__name__)

                timer = LatencyTimer()
                timer.start()

                try:
                    # Execute the pipeline
                    result = func(*args, **kwargs)

                    # Measure and track pipeline latency
                    latency = timer.stop()
                    latency_ms = latency * 1000  # Convert to milliseconds
                    span.set_attribute("pipeline.latency_ms", latency_ms)

                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result

                except Exception as e:
                    timer.stop()
                    # Error handling will be done by child spans
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise

        return wrapper

    return decorator
