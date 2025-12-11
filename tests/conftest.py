"""
Pytest configuration and shared fixtures.
"""

import os
import pytest
from unittest.mock import Mock, patch
from typing import Generator

# Set test environment variables before importing modules
os.environ["PHOENIX_ENDPOINT"] = "https://test-phoenix.example.com"
os.environ["SERVICE_NAME"] = "test-service"
os.environ["OTLP_INSECURE"] = "true"  # Use insecure for testing
os.environ.pop("OPENAI_API_KEY", None)
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)


@pytest.fixture(autouse=True)
def reset_config():
    """Reset configuration before each test."""
    from phoenix_observability.config import reset_config
    reset_config()
    yield
    reset_config()


@pytest.fixture
def mock_tracer():
    """Create a mock OpenTelemetry tracer."""
    from unittest.mock import Mock
    tracer = Mock()
    span = Mock()
    span.__enter__ = Mock(return_value=span)
    span.__exit__ = Mock(return_value=False)
    tracer.start_as_current_span.return_value = span
    return tracer, span


@pytest.fixture
def sample_span():
    """Create a mock OpenTelemetry span."""
    from unittest.mock import Mock
    span = Mock()
    span.set_attribute = Mock()
    span.set_status = Mock()
    span.record_exception = Mock()
    return span


@pytest.fixture
def mock_env(monkeypatch):
    """Fixture to mock environment variables."""
    def _set_env(**kwargs):
        for key, value in kwargs.items():
            monkeypatch.setenv(key, value)
    return _set_env


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment for testing."""
    # Remove all phoenix-related env vars
    env_vars_to_remove = [
        "PHOENIX_ENDPOINT",
        "SERVICE_NAME",
        "OTLP_ENDPOINT",
        "OTLP_INSECURE",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ]
    for var in env_vars_to_remove:
        monkeypatch.delenv(var, raising=False)
    yield
    # Restore defaults
    os.environ["PHOENIX_ENDPOINT"] = "https://test-phoenix.example.com"
    os.environ["SERVICE_NAME"] = "test-service"

