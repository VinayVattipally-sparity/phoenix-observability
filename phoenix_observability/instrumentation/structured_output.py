"""
Structured output tracking utilities.

Tracks JSON parse failures, schema violations, and validation issues.
"""

import json
import logging
from typing import Any, Dict, Optional

from opentelemetry import trace

logger = logging.getLogger(__name__)


def track_json_parse_failure(
    span: trace.Span,
    raw_response: str,
    error: Exception,
) -> None:
    """
    Track JSON parsing failure.

    Args:
        span: OpenTelemetry span
        raw_response: Raw response that failed to parse
        error: Exception that occurred
    """
    span.set_attribute("json.valid", False)
    span.set_attribute("structured_output.parse_failed", True)
    span.set_attribute("structured_output.error_type", type(error).__name__)
    span.set_attribute("json.error_message", str(error))
    span.set_attribute("structured_output.raw_response_length", len(raw_response))

    # Try to extract partial JSON if possible
    try:
        # Look for JSON-like content
        if "{" in raw_response or "[" in raw_response:
            span.set_attribute("structured_output.has_json_like_content", True)
    except:
        pass


def track_schema_violation(
    span: trace.Span,
    expected_schema: Dict[str, Any],
    actual_data: Dict[str, Any],
    violations: Optional[list] = None,
) -> None:
    """
    Track schema validation violations.

    Args:
        span: OpenTelemetry span
        expected_schema: Expected schema structure
        actual_data: Actual data received
        violations: List of violation messages
    """
    span.set_attribute("schema.validation_failed", True)
    span.set_attribute("structured_output.schema_violation", True)
    span.set_attribute("structured_output.expected_keys", list(expected_schema.keys()))
    span.set_attribute("structured_output.actual_keys", list(actual_data.keys()))

    missing_keys = set(expected_schema.keys()) - set(actual_data.keys())
    if missing_keys:
        span.set_attribute("schema.missing_keys", list(missing_keys))
        span.set_attribute("structured_output.missing_keys", list(missing_keys))

    extra_keys = set(actual_data.keys()) - set(expected_schema.keys())
    if extra_keys:
        span.set_attribute("structured_output.extra_keys", list(extra_keys))

    if violations:
        span.set_attribute("structured_output.violations", str(violations))


def track_validation_error(
    span: trace.Span,
    field: str,
    error: Exception,
    value: Any = None,
) -> None:
    """
    Track field validation error.

    Args:
        span: OpenTelemetry span
        field: Field name that failed validation
        error: Validation error
        value: Value that failed validation (optional)
    """
    span.set_attribute("structured_output.validation_failed", True)
    span.set_attribute("structured_output.validation_field", field)
    span.set_attribute("structured_output.validation_error", str(error))

    if value is not None:
        span.set_attribute(
            "structured_output.validation_value",
            str(value)[:500],  # Truncate long values
        )


def parse_and_validate_json(
    span: trace.Span,
    response: str,
    schema: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parse JSON and validate against schema if provided.

    Args:
        span: OpenTelemetry span
        response: JSON string to parse
        schema: Optional schema to validate against

    Returns:
        Parsed JSON dict or None if parsing/validation failed
    """
    try:
        data = json.loads(response)
        span.set_attribute("structured_output.parse_success", True)

        if schema:
            violations = []
            for key, expected_type in schema.items():
                if key not in data:
                    violations.append(f"Missing required key: {key}")
                elif not isinstance(data[key], expected_type):
                    violations.append(
                        f"Type mismatch for {key}: expected {expected_type}, "
                        f"got {type(data[key])}"
                    )

            if violations:
                track_schema_violation(span, schema, data, violations)
                span.set_attribute("json.valid", False)
                return None

            span.set_attribute("structured_output.validation_success", True)
            span.set_attribute("json.valid", True)

        span.set_attribute("json.valid", True)
        return data

    except json.JSONDecodeError as e:
        track_json_parse_failure(span, response, e)
        span.set_attribute("json.valid", False)
        return None
    except Exception as e:
        span.set_attribute("json.valid", False)
        span.set_attribute("structured_output.unexpected_error", True)
        span.set_attribute("json.error_message", str(e))
        span.set_attribute("structured_output.error_message", str(e))
        logger.error(f"Unexpected error in JSON parsing: {e}", exc_info=True)
        return None

