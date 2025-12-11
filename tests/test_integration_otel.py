"""
Integration tests for OpenTelemetry setup.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from phoenix_observability.otel_setup import init_observability, get_tracer


class TestInitObservability:
    """Integration tests for init_observability function."""

    @patch('phoenix_observability.otel_setup.HttpExporter')
    @patch('phoenix_observability.otel_setup.BatchSpanProcessor')
    @patch('phoenix_observability.otel_setup.TracerProvider')
    @patch('phoenix_observability.otel_setup.Resource')
    def test_init_observability_http(self, mock_resource, mock_tracer_provider, mock_processor, mock_exporter):
        """Test initialization with HTTP endpoint."""
        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance
        
        init_observability(
            service_name="test-service",
            project_name="test-project",
            phoenix_endpoint="https://phoenix.example.com",
        )
        
        # Check that exporter was created
        assert mock_exporter.called
        
        # Check that processor was created
        assert mock_processor.called
        
        # Check that provider was configured
        assert mock_tracer_provider.called

    @patch('phoenix_observability.otel_setup.GrpcExporter')
    @patch('phoenix_observability.otel_setup.BatchSpanProcessor')
    @patch('phoenix_observability.otel_setup.TracerProvider')
    @patch('phoenix_observability.otel_setup.trace')
    def test_init_observability_grpc(self, mock_trace, mock_tracer_provider, mock_processor, mock_exporter):
        """Test initialization with gRPC endpoint."""
        mock_provider_instance = Mock()
        mock_tracer_provider.return_value = mock_provider_instance
        
        # Mock trace.set_tracer_provider to avoid "Overriding not allowed" warning
        mock_trace.set_tracer_provider = Mock()
        
        init_observability(
            service_name="test-service",
            phoenix_endpoint="grpc://phoenix.example.com:4317",
        )
        
        # Check that gRPC exporter was created with correct endpoint
        assert mock_exporter.called
        # Verify it was called with the grpc endpoint (without /v1/traces)
        call_args = mock_exporter.call_args
        assert call_args is not None
        assert call_args[1]['endpoint'] == "grpc://phoenix.example.com:4317"

    def test_init_observability_sanitizes_inputs(self):
        """Test that inputs are sanitized."""
        with pytest.raises(ValueError):
            # Invalid URL should raise error
            init_observability(
                service_name="test-service",
                phoenix_endpoint="invalid-url",
            )

    def test_init_observability_sanitizes_service_name(self):
        """Test that service name is sanitized."""
        with pytest.raises(ValueError):
            # Invalid service name should raise error
            init_observability(
                service_name="<script>alert('xss')</script>",
                phoenix_endpoint="https://phoenix.example.com",
            )


class TestGetTracer:
    """Integration tests for get_tracer function."""

    def test_get_tracer_default(self):
        """Test getting tracer with default name."""
        from phoenix_observability.config import reset_config
        reset_config()
        
        tracer = get_tracer()
        assert tracer is not None

    def test_get_tracer_custom_name(self):
        """Test getting tracer with custom name."""
        from phoenix_observability.config import reset_config
        reset_config()
        
        tracer = get_tracer("custom-tracer")
        assert tracer is not None

    def test_get_tracer_invalid_name_type(self):
        """Test getting tracer with invalid name type."""
        from phoenix_observability.config import reset_config
        reset_config()
        
        with pytest.raises(TypeError):
            get_tracer(123)  # Should be string

    def test_get_tracer_empty_string(self):
        """Test getting tracer with empty string (should use default)."""
        from phoenix_observability.config import reset_config
        reset_config()
        
        tracer = get_tracer("")
        assert tracer is not None

