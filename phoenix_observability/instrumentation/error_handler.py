"""
Error handling utilities for spans.

Handles exceptions and attaches metadata to spans.
"""

import logging
import traceback
from typing import Any, Optional

from opentelemetry import trace

from phoenix_observability.utils.security import (
    sanitize_error_message,
    sanitize_stack_trace,
)

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

    # Sanitize error message to remove sensitive data
    sanitized_msg = sanitize_error_message(str(exception), exception)

    # Set error status
    span.set_status(trace.Status(trace.StatusCode.ERROR, sanitized_msg))

    # Add error attributes (with sanitized message)
    span.set_attribute("error.type", type(exception).__name__)
    span.set_attribute("error.message", sanitized_msg)
    span.set_attribute("error.is_failure", True)

    if include_stack_trace:
        stack_trace = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
        sanitized_trace = sanitize_stack_trace(stack_trace)
        span.set_attribute("error.traceback", sanitized_trace)


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
            # Sanitize context values to prevent sensitive data leakage
            sanitized_value = sanitize_error_message(str(value))
            span.set_attribute(f"error.context.{key}", sanitized_value)

    # Sanitize error message before logging
    sanitized_msg = sanitize_error_message(str(exception), exception)
    logger.error(f"Error in span {span.name}: {sanitized_msg}", exc_info=False)  # Don't include full trace in logs

