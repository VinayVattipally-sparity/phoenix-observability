"""
Span enrichment utilities.

Adds contextual metadata such as user ID, request ID, session ID, etc.
"""

import logging
from typing import Dict, Optional

from opentelemetry import trace

logger = logging.getLogger(__name__)


def enrich_span_with_context(
    span: trace.Span,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    custom_context: Optional[Dict[str, str]] = None,
) -> None:
    """
    Enrich a span with contextual metadata.

    Args:
        span: OpenTelemetry span
        user_id: User identifier
        request_id: Request identifier
        session_id: Session identifier
        correlation_id: Correlation identifier for distributed tracing
        custom_context: Custom context dictionary
    """
    if user_id:
        span.set_attribute("user.id", str(user_id))

    if request_id:
        span.set_attribute("request.id", str(request_id))

    if session_id:
        span.set_attribute("session.id", str(session_id))

    if correlation_id:
        span.set_attribute("correlation.id", str(correlation_id))

    if custom_context:
        for key, value in custom_context.items():
            span.set_attribute(f"context.{key}", str(value))


def get_current_span() -> Optional[trace.Span]:
    """
    Get the current active span.

    Returns:
        Current span or None if no active span
    """
    return trace.get_current_span()


def enrich_current_span(
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    custom_context: Optional[Dict[str, str]] = None,
) -> None:
    """
    Enrich the current active span with contextual metadata.

    Args:
        user_id: User identifier
        request_id: Request identifier
        session_id: Session identifier
        correlation_id: Correlation identifier
        custom_context: Custom context dictionary
    """
    span = get_current_span()
    if span:
        enrich_span_with_context(
            span, user_id, request_id, session_id, correlation_id, custom_context
        )
    else:
        logger.warning("No active span found for enrichment")

