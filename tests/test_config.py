"""
Unit tests for config.py module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from phoenix_observability.config import ObservabilityConfig, get_config, reset_config


class TestObservabilityConfig:
    """Tests for ObservabilityConfig class."""

    def test_default_config(self, clean_env):
        """Test default configuration values."""
        config = ObservabilityConfig()
        assert config.phoenix_endpoint == "https://sparity-phoenix.com"
        assert config.environment == "dev"
        assert config.default_service_name == "phoenix_observability"
        assert config.otlp_insecure is False  # Should default to secure

    def test_config_from_env(self, mock_env):
        """Test configuration from environment variables."""
        mock_env(
            PHOENIX_ENDPOINT="https://custom-phoenix.com",
            ENVIRONMENT="production",
            SERVICE_NAME="my-service",
            OTLP_INSECURE="true",
        )
        reset_config()
        config = ObservabilityConfig()
        assert config.phoenix_endpoint == "https://custom-phoenix.com"
        assert config.environment == "production"
        assert config.default_service_name == "my-service"
        assert config.otlp_insecure is True

    def test_otlp_endpoint_default(self):
        """Test OTLP endpoint defaults to Phoenix endpoint."""
        config = ObservabilityConfig()
        assert config.otlp_endpoint == f"{config.phoenix_endpoint}/v1/traces"

    def test_otlp_endpoint_custom(self, mock_env):
        """Test custom OTLP endpoint."""
        mock_env(OTLP_ENDPOINT="https://custom-otlp.com/v1/traces")
        reset_config()
        config = ObservabilityConfig()
        assert config.otlp_endpoint == "https://custom-otlp.com/v1/traces"

    def test_feature_flags(self, mock_env):
        """Test feature flag configuration."""
        mock_env(
            ENABLE_GPU_TRACKING="true",
            ENABLE_PII_TRACKING="false",
            ENABLE_COST_TRACKING="true",
        )
        reset_config()
        config = ObservabilityConfig()
        assert config.enable_gpu_tracking is True
        assert config.enable_pii_tracking is False
        assert config.enable_cost_tracking is True

    def test_batch_settings(self, mock_env):
        """Test batch processor settings."""
        mock_env(
            BATCH_TIMEOUT_MS="10000",
            MAX_EXPORT_BATCH_SIZE="1024",
        )
        reset_config()
        config = ObservabilityConfig()
        assert config.batch_timeout_ms == 10000
        assert config.max_export_batch_size == 1024

    def test_sanitization_settings(self, mock_env):
        """Test sanitization length settings."""
        mock_env(
            MAX_PROMPT_LENGTH="5000",
            MAX_RESPONSE_LENGTH="25000",
        )
        reset_config()
        config = ObservabilityConfig()
        assert config.max_prompt_length == 5000
        assert config.max_response_length == 25000


class TestGetConfig:
    """Tests for get_config() function."""

    def test_get_config_singleton(self):
        """Test that get_config returns the same instance."""
        reset_config()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_get_config_thread_safety(self):
        """Test thread-safety of get_config."""
        import threading
        
        reset_config()
        configs = []
        
        def get_config_in_thread():
            configs.append(get_config())
        
        threads = [threading.Thread(target=get_config_in_thread) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        # All should be the same instance
        assert all(c is configs[0] for c in configs)

    def test_reset_config(self):
        """Test reset_config functionality."""
        config1 = get_config()
        reset_config()
        config2 = get_config()
        # Should be different instances after reset
        assert config1 is not config2

