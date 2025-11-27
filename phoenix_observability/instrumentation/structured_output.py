"""
Structured output tracking utilities.

Tracks JSON parse failures, schema violations, and validation issues.
Supports Pydantic models, JSON Schema, and basic type validation.
"""

import json
import logging
from typing import Any, Dict, Optional, Type, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

from opentelemetry import trace

logger = logging.getLogger(__name__)

# Try importing Pydantic for model validation
try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.debug("Pydantic not available. Pydantic model validation will be skipped.")


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
    schema: Optional[Union[Dict[str, Any], Type[Any]]] = None,
) -> Optional[Union[Dict[str, Any], Any]]:
    """
    Parse JSON and validate against schema or Pydantic model.

    Args:
        span: OpenTelemetry span
        response: JSON string to parse
        schema: Optional schema (dict with type hints) or Pydantic BaseModel class

    Returns:
        Parsed JSON dict, Pydantic model instance, or None if parsing/validation failed
    """
    try:
        data = json.loads(response)
        span.set_attribute("structured_output.parse_success", True)

        if schema:
            # Check if schema is a Pydantic model
            if PYDANTIC_AVAILABLE and isinstance(schema, type) and issubclass(schema, BaseModel):
                return _validate_pydantic_model(span, data, schema)
            elif isinstance(schema, dict):
                # Basic type-based validation
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
            else:
                logger.warning(f"Unknown schema type: {type(schema)}")
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


def _validate_pydantic_model(
    span: trace.Span,
    data: Dict[str, Any],
    model_class: Type[BaseModel]
) -> Optional[BaseModel]:
    """
    Validate data against a Pydantic model.

    Args:
        span: OpenTelemetry span
        data: Parsed JSON data
        model_class: Pydantic BaseModel class

    Returns:
        Validated Pydantic model instance or None if validation failed
    """
    if not PYDANTIC_AVAILABLE:
        logger.warning("Pydantic not available, cannot validate Pydantic model")
        span.set_attribute("json.valid", False)
        span.set_attribute("structured_output.validation_error", "Pydantic not available")
        return None

    try:
        # Validate and create model instance
        model_instance = model_class(**data)
        
        span.set_attribute("structured_output.validation_success", True)
        span.set_attribute("structured_output.validation_type", "pydantic")
        span.set_attribute("structured_output.model_name", model_class.__name__)
        span.set_attribute("json.valid", True)
        
        # Track model fields for observability
        field_names = list(model_class.model_fields.keys())
        span.set_attribute("structured_output.model_fields", field_names)
        span.set_attribute("structured_output.model_field_count", len(field_names))
        
        return model_instance
        
    except ValidationError as e:
        # Track Pydantic validation errors
        span.set_attribute("json.valid", False)
        span.set_attribute("structured_output.validation_failed", True)
        span.set_attribute("structured_output.validation_type", "pydantic")
        span.set_attribute("structured_output.model_name", model_class.__name__)
        span.set_attribute("structured_output.validation_error", "PydanticValidationError")
        
        # Track detailed validation errors
        errors = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            error_msg = f"{field}: {error['msg']}"
            errors.append(error_msg)
            
            # Track individual field errors
            span.set_attribute(
                f"structured_output.field_error.{field}",
                error["msg"][:200]  # Truncate long messages
            )
        
        span.set_attribute("structured_output.validation_errors", str(errors))
        span.set_attribute("structured_output.validation_error_count", len(errors))
        
        logger.warning(f"Pydantic validation failed for {model_class.__name__}: {errors}")
        return None
        
    except Exception as e:
        span.set_attribute("json.valid", False)
        span.set_attribute("structured_output.unexpected_error", True)
        span.set_attribute("structured_output.error_message", str(e))
        logger.error(f"Unexpected error in Pydantic validation: {e}", exc_info=True)
        return None


def validate_structured_output(
    span: trace.Span,
    response: Union[str, Dict[str, Any]],
    expected_schema: Optional[Union[Dict[str, Any], Type[Any]]] = None,
) -> Optional[Union[Dict[str, Any], Any]]:
    """
    Comprehensive structured output validation.

    Supports:
    - JSON string parsing
    - Pydantic model validation
    - Basic type-based schema validation
    - Parse failure tracking

    Args:
        span: OpenTelemetry span
        response: JSON string or already-parsed dict
        expected_schema: Optional schema (dict) or Pydantic BaseModel class

    Returns:
        Validated data (dict or Pydantic model) or None if validation failed
    """
    # If response is already a dict, validate directly
    if isinstance(response, dict):
        if expected_schema:
            if PYDANTIC_AVAILABLE and isinstance(expected_schema, type) and issubclass(expected_schema, BaseModel):
                return _validate_pydantic_model(span, response, expected_schema)
            elif isinstance(expected_schema, dict):
                # Basic validation
                violations = []
                for key, expected_type in expected_schema.items():
                    if key not in response:
                        violations.append(f"Missing required key: {key}")
                    elif not isinstance(response[key], expected_type):
                        violations.append(
                            f"Type mismatch for {key}: expected {expected_type}, "
                            f"got {type(response[key])}"
                        )
                
                if violations:
                    track_schema_violation(span, expected_schema, response, violations)
                    span.set_attribute("json.valid", False)
                    return None
                
                span.set_attribute("structured_output.validation_success", True)
                span.set_attribute("json.valid", True)
        
        span.set_attribute("json.valid", True)
        return response
    
    # If response is a string, parse and validate
    if isinstance(response, str):
        return parse_and_validate_json(span, response, expected_schema)
    
    # Unknown type
    span.set_attribute("json.valid", False)
    span.set_attribute("structured_output.unexpected_type", type(response).__name__)
    logger.warning(f"Unexpected response type: {type(response)}")
    return None

