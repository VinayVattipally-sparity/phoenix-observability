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

logger = logging.getLogger(__name__)


def init_observability(
    service_name: Optional[str] = None,
    phoenix_endpoint: Optional[str] = None,
    environment: Optional[str] = None,
) -> None:
    """
    Initialize OpenTelemetry tracing for the project.

    Args:
        service_name: Name of the service (overrides config default)
        phoenix_endpoint: Phoenix endpoint URL (overrides config)
        environment: Environment name (dev/stage/prod, overrides config)
    """
    config = get_config()

    # Override config with provided parameters
    if service_name is None:
        service_name = config.default_service_name
    if phoenix_endpoint is None:
        phoenix_endpoint = config.phoenix_endpoint
    if environment is None:
        environment = config.environment

    # Create resource with service metadata
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.1.0",
            "deployment.environment": environment,
        }
    )

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

    logger.info(
        f"OpenTelemetry initialized for service '{service_name}' "
        f"with Phoenix endpoint: {phoenix_endpoint}"
    )


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

