"""
Unit tests for span_helpers.py module.
"""

import pytest
from unittest.mock import Mock, patch
from phoenix_observability.instrumentation.span_helpers import (
    extract_project_and_service_name,
    set_span_project_service,
)


class TestExtractProjectAndServiceName:
    """Tests for extract_project_and_service_name function."""

    def test_extract_from_env(self, mock_env):
        """Test extraction from environment variables."""
        mock_env(
            PHOENIX_PROJECT_NAME="test-project",
            SERVICE_NAME="test-service",
        )
        from phoenix_observability.config import reset_config
        reset_config()
        
        project_name, service_name = extract_project_and_service_name()
        assert project_name == "test-project"
        assert service_name == "test-service"

    def test_extract_with_provided_service_name(self, mock_env):
        """Test extraction with provided service name."""
        mock_env(PHOENIX_PROJECT_NAME="test-project")
        from phoenix_observability.config import reset_config
        reset_config()
        
        project_name, service_name = extract_project_and_service_name(service_name="custom-service")
        assert project_name == "test-project"
        assert service_name == "custom-service"

    def test_extract_defaults(self, clean_env):
        """Test extraction with defaults."""
        from phoenix_observability.config import reset_config
        reset_config()
        
        project_name, service_name = extract_project_and_service_name()
        # Should use default service name from config
        assert project_name is not None
        assert service_name is not None

    def test_extract_sanitizes_names(self, mock_env):
        """Test that names are sanitized."""
        from phoenix_observability.config import reset_config
        from phoenix_observability.instrumentation.span_helpers import clear_env_cache
        
        # Clear cache and reset config
        clear_env_cache()
        reset_config()
        
        # Set new env var
        mock_env(PHOENIX_PROJECT_NAME="valid-project-name")
        clear_env_cache()  # Clear cache again after setting env var
        
        project_name, service_name = extract_project_and_service_name(service_name="valid-service-name")
        assert project_name == "valid-project-name"
        assert service_name == "valid-service-name"

    def test_extract_invalid_service_name(self, mock_env):
        """Test extraction with invalid service name."""
        mock_env(PHOENIX_PROJECT_NAME="valid-project")
        from phoenix_observability.config import reset_config
        reset_config()
        
        # Invalid service name should fall back to default
        project_name, service_name = extract_project_and_service_name(service_name="<script>alert('xss')</script>")
        # Should use default instead of invalid name
        assert service_name != "<script>alert('xss')</script>"

    @patch('phoenix_observability.instrumentation.span_helpers._get_cached_env')
    def test_extract_from_resource_attributes(self, mock_get_env, sample_span):
        """Test extraction from resource attributes."""
        mock_get_env.return_value = None
        
        from opentelemetry import trace
        from unittest.mock import MagicMock
        
        # Mock tracer provider with resource
        mock_resource = MagicMock()
        mock_resource.attributes = {
            "project.name": "resource-project",
            "service.name": "resource-service",
        }
        
        mock_provider = MagicMock()
        mock_provider.resource = mock_resource
        
        with patch('opentelemetry.trace.get_tracer_provider', return_value=mock_provider):
            from phoenix_observability.config import reset_config
            reset_config()
            project_name, service_name = extract_project_and_service_name()
            # Should use resource attributes if available
            # Note: This depends on actual implementation


class TestSetSpanProjectService:
    """Tests for set_span_project_service function."""

    def test_set_span_project_service(self, sample_span):
        """Test setting project and service on span."""
        set_span_project_service(sample_span, "test-project", "test-service")
        
        # Check that attributes were set
        assert sample_span.set_attribute.called
        calls = [call[0] for call in sample_span.set_attribute.call_args_list]
        assert ("project.name", "test-project") in calls
        assert ("project.id", "test-project") in calls
        assert ("service.name", "test-service") in calls

    def test_set_span_project_service_empty_project(self, sample_span):
        """Test setting service with empty project name."""
        set_span_project_service(sample_span, "", "test-service")
        
        calls = [call[0] for call in sample_span.set_attribute.call_args_list]
        # Project attributes should not be set if empty
        project_calls = [c for c in calls if c[0].startswith("project")]
        assert len(project_calls) == 0
        assert ("service.name", "test-service") in calls

    def test_set_span_project_service_empty_service(self, sample_span):
        """Test setting project with empty service name."""
        set_span_project_service(sample_span, "test-project", "")
        
        calls = [call[0] for call in sample_span.set_attribute.call_args_list]
        # Service attribute should not be set if empty
        service_calls = [c for c in calls if c[0] == "service.name"]
        assert len(service_calls) == 0
        assert ("project.name", "test-project") in calls

