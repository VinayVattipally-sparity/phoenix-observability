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
) -> Callable[[Callable], Callable]:
    """
    Decorator to instrument complete pipelines for end-to-end observability.

    This decorator creates high-level spans that track the complete execution of multi-stage
    pipelines (e.g., RAG pipelines, agent workflows, data processing pipelines). It measures
    total pipeline latency and can be used in combination with other instrumentation decorators
    to create nested spans.

    Args:
        pipeline_name: Name identifier for this pipeline (e.g., "rag_pipeline", "data_processing").
            If not provided, defaults to the decorated function's name. Used in span names
            and attributes for identification. Should be descriptive to help identify the pipeline
            in observability dashboards.

    Returns:
        A decorator function that wraps the original function with observability instrumentation.
        The wrapped function maintains the same signature and return value as the original.
        The span tracks the total execution time as `pipeline.latency_ms`.

    Example:
        Basic usage::

            @instrument_pipeline(pipeline_name="rag_pipeline")
            def complete_rag_pipeline(query: str):
                docs = retrieve_documents(query)
                context = format_context(docs)
                response = generate_response(context, query)
                return response

        With nested instrumentation::

            @instrument_pipeline(pipeline_name="multi_stage")
            def multi_stage_pipeline(input_data):
                # This creates a parent span
                stage1_result = stage1(input_data)  # Can have its own @instrument_llm
                stage2_result = stage2(stage1_result)  # Can have its own instrumentation
                return stage2_result
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
