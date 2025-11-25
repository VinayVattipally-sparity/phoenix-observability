"""
Span utility functions.

Utilities for setting nested span attributes consistently.
"""

import logging
from typing import Any, Dict

from opentelemetry import trace

logger = logging.getLogger(__name__)


def set_nested_attribute(
    span: trace.Span,
    key: str,
    value: Any,
    max_depth: int = 3,
    max_length: int = 1000,
) -> None:
    """
    Set a nested attribute on a span, flattening nested structures.

    Args:
        span: OpenTelemetry span
        key: Base key name
        value: Value to set (can be dict, list, or primitive)
        max_depth: Maximum nesting depth
        max_length: Maximum string length for values
    """
    if isinstance(value, dict):
        if max_depth <= 0:
            span.set_attribute(key, str(value)[:max_length])
            return

        for nested_key, nested_value in value.items():
            nested_key_full = f"{key}.{nested_key}"
            set_nested_attribute(
                span, nested_key_full, nested_value, max_depth - 1, max_length
            )

    elif isinstance(value, list):
        span.set_attribute(f"{key}.count", len(value))
        for i, item in enumerate(value[:10]):  # Limit to first 10 items
            set_nested_attribute(
                span, f"{key}.{i}", item, max_depth - 1, max_length
            )

    else:
        # Primitive value
        value_str = str(value)
        if len(value_str) > max_length:
            value_str = value_str[:max_length] + "... [truncated]"
        span.set_attribute(key, value_str)


def set_span_attributes(
    span: trace.Span,
    attributes: Dict[str, Any],
    prefix: str = "",
    max_length: int = 1000,
) -> None:
    """
    Set multiple attributes on a span with optional prefix.

    Args:
        span: OpenTelemetry span
        attributes: Dictionary of attributes to set
        prefix: Optional prefix for all keys
        max_length: Maximum string length for values
    """
    for key, value in attributes.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, (dict, list)):
            set_nested_attribute(span, full_key, value, max_length=max_length)
        else:
            value_str = str(value)
            if len(value_str) > max_length:
                value_str = value_str[:max_length] + "... [truncated]"
            span.set_attribute(full_key, value_str)

