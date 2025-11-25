"""
Error handling utilities for spans.

Handles exceptions and attaches metadata to spans.
"""

import logging
import traceback
from typing import Any, Optional

from opentelemetry import trace

logger = logging.getLogger(__name__)


def attach_exception_to_span(
    span: trace.Span,
    exception: Exception,
    include_stack_trace: bool = True,
) -> None:
    """
    Attach exception information to a span.

    Args:
        span: OpenTelemetry span
        exception: Exception that occurred
        include_stack_trace: Whether to include full stack trace
    """
    span.record_exception(exception)

    # Set error status
    span.set_status(trace.Status(trace.StatusCode.ERROR, str(exception)))

    # Add error attributes
    span.set_attribute("error.type", type(exception).__name__)
    span.set_attribute("error.message", str(exception))
    span.set_attribute("error.is_failure", True)

    if include_stack_trace:
        stack_trace = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        span.set_attribute("error.traceback", stack_trace)


def handle_error(
    span: trace.Span,
    exception: Exception,
    context: Optional[dict] = None,
) -> None:
    """
    Comprehensive error handling for spans.

    Args:
        span: OpenTelemetry span
        exception: Exception that occurred
        context: Optional context dictionary to attach
    """
    attach_exception_to_span(span, exception)

    if context:
        for key, value in context.items():
            span.set_attribute(f"error.context.{key}", str(value))

    logger.error(f"Error in span {span.name}: {exception}", exc_info=True)

