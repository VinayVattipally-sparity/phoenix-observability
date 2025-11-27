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

from phoenix_observability.config import get_config  # type: ignore

logger = logging.getLogger(__name__)


def create_phoenix_project(
    project_name: str,
    phoenix_endpoint: str,
    description: Optional[str] = None,
) -> bool:
    """
    Create a Phoenix project programmatically.
    
    Tries to use Phoenix client API if available, otherwise uses HTTP API directly.
    
    Args:
        project_name: Name of the project to create
        phoenix_endpoint: Phoenix server endpoint URL
        description: Optional project description
        
    Returns:
        True if project was created or already exists, False on error
    """
    try:
        # Try using Phoenix client API (preferred method)
        try:
            from phoenix.client import Client  # type: ignore
            
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
        service_name: Name of the service (overrides config default)
        project_name: Name of the project (used to organize traces in Phoenix)
        phoenix_endpoint: Phoenix endpoint URL (overrides config)
        environment: Environment name (dev/stage/prod, overrides config)
    """
    import os
    
    config = get_config()

    # Override config with provided parameters
    if service_name is None:
        service_name = config.default_service_name
    if phoenix_endpoint is None:
        phoenix_endpoint = config.phoenix_endpoint
    if environment is None:
        environment = config.environment
    
    # Try to use Phoenix's own registration method if available (handles project organization better)
    try:
        from phoenix.otel import register as phoenix_register  # type: ignore
        
        # Get project name from parameter or environment variable (matching Phoenix folder pattern)
        phoenix_project = project_name or os.getenv("PHOENIX_PROJECT_NAME", service_name)
        
        # Ensure endpoint format is correct (add /v1/traces if not present)
        otlp_endpoint = phoenix_endpoint
        if otlp_endpoint and not otlp_endpoint.endswith('/v1/traces'):
            otlp_endpoint = otlp_endpoint.rstrip('/') + '/v1/traces'
        
        logger.info(f"Using Phoenix's register() method with project_name='{phoenix_project}'")
        
        # Use Phoenix's register function which handles project_name properly
        # Matching the pattern from D:/Phoenix/anomaly_detection/tracing.py
        tracer_provider = phoenix_register(
            endpoint=otlp_endpoint,
            project_name=phoenix_project,
            protocol="http/protobuf",  # Use HTTP/protobuf for better compatibility
            batch=True,
            verbose=True
        )
        
        log_msg = f"Phoenix OpenTelemetry initialized for service '{service_name}'"
        if phoenix_project:
            log_msg += f" in project '{phoenix_project}'"
        log_msg += f" with Phoenix endpoint: {otlp_endpoint}"
        logger.info(log_msg)
        return
        
    except ImportError:
        # Phoenix's register not available, use pure OpenTelemetry
        logger.warning(
            "Phoenix's register() not available. "
            "For proper project organization, install: pip install arize-phoenix"
        )
        logger.info("Falling back to pure OpenTelemetry setup")
        pass
    
    # Fallback to pure OpenTelemetry setup
    # IMPORTANT: For Phoenix to organize traces by project with pure OpenTelemetry,
    # we need to set PHOENIX_PROJECT_NAME before tracer provider creation
    # AND use service.name that matches the project name
    if project_name:
        # Set environment variable (Phoenix reads this)
        os.environ["PHOENIX_PROJECT_NAME"] = project_name
        # Also set PHOENIX_COLLECTOR_ENDPOINT if not already set
        if not os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
            os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint
        logger.info(f"Set PHOENIX_PROJECT_NAME={project_name} for project organization")

    # Build resource attributes
    # NOTE: When using pure OpenTelemetry, Phoenix may use service.name as project identifier
    # So we set service.name to project_name if provided, otherwise use service_name
    resource_attributes = {
        "service.name": project_name if project_name else service_name,
        "service.version": "0.1.0",
        "deployment.environment": environment,
    }
    
    # Add project name if provided (Phoenix uses this to organize traces)
    if project_name:
        # Set as resource attributes for OpenTelemetry
        resource_attributes["project.name"] = project_name
        resource_attributes["project.id"] = project_name
        # Phoenix-specific attributes (Phoenix may read these)
        resource_attributes["phoenix.project.name"] = project_name
        resource_attributes["phoenix.project.id"] = project_name
        # Also keep original service name for reference
        resource_attributes["service.instance.id"] = service_name
        
        # Create the project in Phoenix if it doesn't exist
        # Note: Projects are also created automatically when traces with project.name are sent
        project_created = create_phoenix_project(
            project_name=project_name,
            phoenix_endpoint=phoenix_endpoint,
            description=f"Project for {service_name} service"
        )
        if not project_created:
            logger.info(
                f"Project '{project_name}' will be created automatically when traces are sent. "
                f"To create it explicitly, install: pip install arize-phoenix-client"
            )

    # Create resource with service metadata
    resource = Resource.create(resource_attributes)

    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)

    # Configure OTLP exporter
    otlp_endpoint = config.otlp_endpoint
    if phoenix_endpoint and not config.otlp_endpoint:
        # Construct OTLP endpoint from Phoenix endpoint
        otlp_endpoint = f"{phoenix_endpoint}/v1/traces"

    # Determine which exporter to use based on endpoint protocol
    # Use HTTP exporter if endpoint starts with http:// or https://
    # Otherwise default to gRPC
    use_http = otlp_endpoint.startswith("http://") or otlp_endpoint.startswith("https://")
    
    if use_http:
        # Use HTTP exporter for HTTP/HTTPS endpoints
        # HTTP exporter doesn't support 'insecure' parameter
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
        max_queue_size=2048,
        export_timeout_millis=config.batch_timeout_ms,
        max_export_batch_size=config.max_export_batch_size,
    )

    # Add processor to tracer provider
    tracer_provider.add_span_processor(span_processor)

    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)

    log_msg = f"OpenTelemetry initialized for service '{service_name}'"
    if project_name:
        log_msg += f" in project '{project_name}'"
    log_msg += f" with Phoenix endpoint: {phoenix_endpoint}"
    logger.info(log_msg)


def get_tracer(name: str = None):
    """
    Get a tracer instance.

    Args:
        name: Name of the tracer (defaults to service name)

    Returns:
        Tracer instance
    """
    config = get_config()
    tracer_name = name or config.default_service_name
    return trace.get_tracer(tracer_name)

