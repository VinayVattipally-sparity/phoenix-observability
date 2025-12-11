"""
Shared helper functions for instrumentation wrappers.

This module contains common utilities used across all instrumentation wrappers
to avoid code duplication.
"""

import logging
import os
from typing import Optional, Tuple

from opentelemetry import trace

logger = logging.getLogger(__name__)

# Cache for environment variables to avoid repeated reads
import threading
_env_cache: dict = {}
_cache_lock = threading.Lock()

def _get_cached_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with caching.
    
    Args:
        key: Environment variable name.
        default: Default value if not found.
        
    Returns:
        Environment variable value or default.
    """
    global _env_cache
    
    # Check cache first (fast path without lock)
    if key in _env_cache:
        return _env_cache[key]
    
    # Read from environment and cache
    value = os.getenv(key, default)
    
    # Cache the value (even if None)
    with _cache_lock:
        # Double-check after acquiring lock
        if key not in _env_cache:
            _env_cache[key] = value
        else:
            value = _env_cache[key]
    
    return value


def clear_env_cache():
    """
    Clear the environment variable cache.
    
    Useful for testing or when environment variables change at runtime.
    """
    global _env_cache
    with _cache_lock:
        _env_cache.clear()


def extract_project_and_service_name(
    service_name: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Extract project name and service name from environment or resource attributes.

    This function avoids circular imports by importing config lazily.
    All names are sanitized to prevent injection attacks.

    Args:
        service_name: Optional service name from decorator parameter (will be sanitized).

    Returns:
        Tuple of (project_name, final_service_name).
        
    Raises:
        ValueError: If service_name contains invalid characters
    """
    # Lazy import to avoid circular dependencies
    from phoenix_observability.config import get_config
    from phoenix_observability.utils.security import sanitize_name
    
    config = get_config()
    # Use cached environment variable read
    project_name = _get_cached_env("PHOENIX_PROJECT_NAME") or config.default_service_name
    
    # Sanitize project name from environment
    try:
        project_name = sanitize_name(project_name)
    except ValueError as e:
        logger.warning(f"Invalid project name from environment, using default: {e}")
        project_name = config.default_service_name
    
    final_service_name = service_name
    if final_service_name:
        # Sanitize user-provided service name
        try:
            final_service_name = sanitize_name(final_service_name)
        except ValueError as e:
            logger.warning(f"Invalid service name provided, using default: {e}")
            final_service_name = None

    # Try to get from resource attributes if available
    try:
        from opentelemetry import trace as otel_trace
        tracer_provider = otel_trace.get_tracer_provider()
        if hasattr(tracer_provider, 'resource') and tracer_provider.resource:
            resource_attrs = dict(tracer_provider.resource.attributes)
            resource_project = (
                resource_attrs.get("project.name") 
                or resource_attrs.get("project.id")
            )
            if resource_project:
                try:
                    project_name = sanitize_name(resource_project)
                except ValueError:
                    # Keep existing project_name if resource project name is invalid
                    pass
            
            if not final_service_name:
                resource_service = resource_attrs.get("service.name")
                if resource_service:
                    try:
                        final_service_name = sanitize_name(resource_service)
                    except ValueError:
                        # Will fall back to default below
                        final_service_name = None
    except (AttributeError, RuntimeError, TypeError) as e:
        logger.debug(f"Could not extract resource attributes: {e}")

    # Fallback to config default if still not set
    if not final_service_name:
        final_service_name = config.default_service_name
        # Sanitize default service name
        try:
            final_service_name = sanitize_name(final_service_name)
        except ValueError:
            # If even default is invalid, use a safe fallback
            final_service_name = "phoenix_observability"

    return project_name, final_service_name


def set_span_project_service(
    span: trace.Span,
    project_name: str,
    service_name: str,
) -> None:
    """
    Set project and service attributes on a span.

    Args:
        span: OpenTelemetry span to set attributes on.
        project_name: Project name.
        service_name: Service name.
    """
    if project_name:
        span.set_attribute("project.name", project_name)
        span.set_attribute("project.id", project_name)
    if service_name:
        span.set_attribute("service.name", service_name)

