"""
OpenTelemetry setup and initialization.

Configures OTLP exporter to send traces to Phoenix.
"""

import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

from phoenix_observability.config import get_config
from phoenix_observability.utils.security import sanitize_url, sanitize_name

logger = logging.getLogger(__name__)

# Constants
# Queue size is now configurable via config.max_queue_size


def create_phoenix_project(
    project_name: str,
    phoenix_endpoint: str,
    description: Optional[str] = None,
) -> bool:
    """
    Create a Phoenix project programmatically.
    
    Tries to use Phoenix client API if available, otherwise uses HTTP API directly.
    
    Args:
        project_name: Name of the project to create (will be sanitized)
        phoenix_endpoint: Phoenix server endpoint URL (will be sanitized)
        description: Optional project description
        
    Returns:
        True if project was created or already exists, False on error
        
    Raises:
        ValueError: If project_name or phoenix_endpoint are invalid
    """
    try:
        # Sanitize inputs for security
        project_name = sanitize_name(project_name)
        phoenix_endpoint = sanitize_url(phoenix_endpoint)
        # Try using Phoenix client API (preferred method)
        try:
            from phoenix.client import Client
            
            # Initialize client
            client = Client(base_url=phoenix_endpoint)
            
            # Check if project already exists
            try:
                projects = client.projects.list_projects()
                existing_project = next(
                    (p for p in projects if p.name == project_name),
                    None
                )
                if existing_project:
                    logger.info(f"Project '{project_name}' already exists (ID: {existing_project.id})")
                    return True
            except Exception:
                # If list_projects fails, try to create anyway
                pass
            
            # Create new project
            new_project = client.projects.create_project(
                name=project_name,
                description=description or f"Project for {project_name}"
            )
            logger.info(f"Created Phoenix project '{new_project.name}' (ID: {new_project.id})")
            return True
            
        except ImportError:
            # Phoenix client not available, try HTTP API directly
            logger.debug("Phoenix client not available, trying HTTP API")
            return _create_project_via_http(project_name, phoenix_endpoint, description)
            
    except Exception as e:
        logger.warning(
            f"Failed to create Phoenix project '{project_name}': {e}. "
            "Traces will still be sent, but project may need to be created manually."
        )
        return False


def _create_project_via_http(
    project_name: str,
    phoenix_endpoint: str,
    description: Optional[str] = None,
) -> bool:
    """
    Create a Phoenix project via HTTP API.
    
    Note: This is a fallback method. For best results, install arize-phoenix-client.
    Projects may also be created automatically when traces with project attributes are sent.
    
    Args:
        project_name: Name of the project
        phoenix_endpoint: Phoenix server endpoint
        description: Optional project description
        
    Returns:
        True if successful, False otherwise
    """
    # Note: Phoenix projects are typically created automatically when traces
    # with project.name attributes are sent. Explicit creation via HTTP API
    # may not be supported in all Phoenix versions.
    logger.debug(
        f"HTTP API project creation not available. "
        f"Project '{project_name}' will be created automatically when traces are sent, "
        f"or install 'arize-phoenix-client' for explicit project creation."
    )
    return False


def init_observability(
    service_name: Optional[str] = None,
    project_name: Optional[str] = None,
    phoenix_endpoint: Optional[str] = None,
    environment: Optional[str] = None,
) -> None:
    """
    Initialize OpenTelemetry tracing for the project.

    Args:
        service_name: Name of the service (overrides config default, will be sanitized)
        project_name: Name of the project (used to organize traces in Phoenix, will be sanitized)
        phoenix_endpoint: Phoenix endpoint URL (overrides config, will be sanitized)
        environment: Environment name (dev/stage/prod, overrides config)
        
    Raises:
        ValueError: If service_name, project_name, or phoenix_endpoint are invalid
    """
    import os
    
    config = get_config()

    # Override config with provided parameters
    if service_name is None:
        service_name = config.default_service_name
    else:
        # Sanitize user-provided service name
        service_name = sanitize_name(service_name)
    
    if phoenix_endpoint is None:
        phoenix_endpoint = config.phoenix_endpoint
    else:
        # Sanitize user-provided endpoint URL (only if http/https, allow grpc:// for gRPC)
        if phoenix_endpoint.startswith(("http://", "https://")):
            phoenix_endpoint = sanitize_url(phoenix_endpoint)
        # For grpc:// endpoints, just validate basic format
        elif phoenix_endpoint.startswith("grpc://"):
            from urllib.parse import urlparse
            parsed = urlparse(phoenix_endpoint)
            if not parsed.netloc:
                raise ValueError("gRPC URL must include a hostname")
        else:
            # Try to sanitize, will raise error if invalid
            phoenix_endpoint = sanitize_url(phoenix_endpoint)
    
    if project_name:
        # Sanitize user-provided project name
        project_name = sanitize_name(project_name)
    
    if environment is None:
        environment = config.environment
    
    # Use pure OpenTelemetry setup (matching reference implementation)
    # The reference does NOT use Phoenix register() - it uses pure OpenTelemetry with HttpExporter
    # This matches rag_mini_app which also uses pure OpenTelemetry
    # NOTE: For project organization, we rely on resource attributes and span attributes
    logger.info("Using pure OpenTelemetry setup (matching reference implementation)")

    # Build resource attributes (matching reference exactly)
    # IMPORTANT: service.name must be set correctly for Phoenix to recognize the service
    resource_attributes = {
        "service.name": service_name,
        "service.version": "0.1.0",
        "deployment.environment": environment,
    }
    
    # Ensure service.name is not overridden (Phoenix may set it to "unknown_service" if not careful)
    if not service_name or service_name == "unknown_service":
        logger.warning(f"Service name is '{service_name}', this may cause issues with Phoenix project organization")
    
    # Add project name if provided (Phoenix uses this to organize traces)
    # Phoenix Cloud may read from multiple attribute names, so set all of them
    if project_name:
        resource_attributes["project.name"] = project_name
        resource_attributes["project.id"] = project_name  # Also set as ID for Phoenix project grouping
        resource_attributes["openinference.project.name"] = project_name  # Phoenix may also read this
        resource_attributes["openinference.project.id"] = project_name

    # Create resource with service metadata
    resource = Resource.create(resource_attributes)

    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)

    # Configure OTLP exporter (matching reference implementation exactly)
    # IMPORTANT: For fallback, use OTEL_EXPORTER_OTLP_ENDPOINT if available (it has the full path)
    # Otherwise construct from phoenix_endpoint parameter (if provided) or config.otlp_endpoint
    # HttpExporter requires full path with /v1/traces
    otlp_endpoint = None
    
    # Check if OTEL_EXPORTER_OTLP_ENDPOINT was restored (from failed register() attempt)
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        logger.info(f"Using OTEL_EXPORTER_OTLP_ENDPOINT from environment: {otlp_endpoint}")
    elif phoenix_endpoint:
        # If phoenix_endpoint parameter was provided, use it (takes precedence over config)
        # For gRPC endpoints, use as-is (gRPC doesn't need /v1/traces path)
        if phoenix_endpoint.startswith("grpc://"):
            otlp_endpoint = phoenix_endpoint
        else:
            # For HTTP endpoints, add /v1/traces path
            otlp_endpoint = f"{phoenix_endpoint.rstrip('/')}/v1/traces"
    elif config.otlp_endpoint:
        otlp_endpoint = config.otlp_endpoint
    
    # Always ensure /v1/traces is present for HTTP endpoints (HttpExporter requires full path)
    if otlp_endpoint and not otlp_endpoint.startswith("grpc://") and not otlp_endpoint.endswith('/v1/traces'):
        otlp_endpoint = f"{otlp_endpoint.rstrip('/')}/v1/traces"
        logger.info(f"Added /v1/traces to endpoint: {otlp_endpoint}")
    
    if not otlp_endpoint:
        logger.error("Could not determine OTLP endpoint. Please set PHOENIX_ENDPOINT or OTLP_ENDPOINT")
        return

    # Determine which exporter to use based on endpoint protocol (matching reference exactly)
    # Use HTTP exporter if endpoint starts with http:// or https://
    # Otherwise default to gRPC
    use_http = otlp_endpoint.startswith("http://") or otlp_endpoint.startswith("https://")
    
    if use_http:
        # Use HTTP exporter for HTTP/HTTPS endpoints (matching reference lines 76-82)
        # HTTP exporter doesn't support 'insecure' parameter
        # HttpExporter automatically reads OTEL_EXPORTER_OTLP_HEADERS from environment if set
        # rag_mini_app sets OTEL_EXPORTER_OTLP_HEADERS before calling init_observability()
        exporter = HttpExporter(
            endpoint=otlp_endpoint,
        )
        logger.info(f"Using HTTP exporter for endpoint: {otlp_endpoint}")
    else:
        # Use gRPC exporter for other endpoints (e.g., grpc://)
        exporter = GrpcExporter(
            endpoint=otlp_endpoint,
            insecure=config.otlp_insecure,
        )
        logger.info(f"Using gRPC exporter for endpoint: {otlp_endpoint}")

    # Create batch processor for performance
    span_processor = BatchSpanProcessor(
        exporter,
        max_queue_size=config.max_queue_size,
        export_timeout_millis=config.batch_timeout_ms,
        max_export_batch_size=config.max_export_batch_size,
    )

    # Add processor to tracer provider
    tracer_provider.add_span_processor(span_processor)

    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Build log message efficiently
    if project_name:
        log_msg = f"OpenTelemetry initialized for service '{service_name}' in project '{project_name}' with Phoenix endpoint: {phoenix_endpoint}"
    else:
        log_msg = f"OpenTelemetry initialized for service '{service_name}' with Phoenix endpoint: {phoenix_endpoint}"
    logger.info(log_msg)


def get_tracer(name: Optional[str] = None) -> trace.Tracer:
    """
    Get a tracer instance.

    Args:
        name: Name of the tracer (defaults to service name). If provided, must be a non-empty string.

    Returns:
        Tracer instance
    """
    config = get_config()
    # Validate name if provided
    if name is not None:
        if not isinstance(name, str):
            raise TypeError(f"Tracer name must be a string, got {type(name).__name__}")
        if not name.strip():
            # Empty string treated as None, use default
            name = None
    tracer_name = name or config.default_service_name
    return trace.get_tracer(tracer_name)

