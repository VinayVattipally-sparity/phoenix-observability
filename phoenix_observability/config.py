"""
Central configuration manager for Phoenix Observability.

Loads configuration from environment variables (typically from .env file).
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ObservabilityConfig:
    """Central configuration for observability settings."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Phoenix endpoint - must be set in .env file
        phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT")
        if not phoenix_endpoint:
            raise ValueError(
                "PHOENIX_ENDPOINT environment variable is required. "
                "Please set it in your .env file."
            )
        self.phoenix_endpoint: str = phoenix_endpoint

        # Environment
        self.environment: str = os.getenv("ENVIRONMENT", "dev")

        # Feature flags
        self.enable_gpu_tracking: bool = os.getenv(
            "ENABLE_GPU_TRACKING", "false"
        ).lower() == "true"

        self.enable_pii_tracking: bool = os.getenv(
            "ENABLE_PII_TRACKING", "true"
        ).lower() == "true"

        self.enable_cost_tracking: bool = os.getenv(
            "ENABLE_COST_TRACKING", "true"
        ).lower() == "true"

        # Service name (can be overridden per project)
        self.default_service_name: str = os.getenv(
            "SERVICE_NAME", "phoenix_observability"
        )

        # OTLP exporter settings
        self.otlp_endpoint: Optional[str] = os.getenv("OTLP_ENDPOINT")
        if not self.otlp_endpoint:
            # Default to Phoenix OTLP endpoint
            self.otlp_endpoint = f"{self.phoenix_endpoint}/v1/traces"

        # Batch processor settings
        self.batch_timeout_ms: int = int(os.getenv("BATCH_TIMEOUT_MS", "5000"))
        self.max_export_batch_size: int = int(
            os.getenv("MAX_EXPORT_BATCH_SIZE", "512")
        )

        # Sanitization settings
        self.max_prompt_length: int = int(os.getenv("MAX_PROMPT_LENGTH", "10000"))
        self.max_response_length: int = int(
            os.getenv("MAX_RESPONSE_LENGTH", "50000")
        )

        # OTLP exporter settings
        self.otlp_insecure: bool = os.getenv(
            "OTLP_INSECURE", "true"
        ).lower() == "true"  # Default to insecure for local development

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"ObservabilityConfig("
            f"phoenix_endpoint={self.phoenix_endpoint}, "
            f"environment={self.environment}, "
            f"gpu_tracking={self.enable_gpu_tracking}, "
            f"pii_tracking={self.enable_pii_tracking}, "
            f"cost_tracking={self.enable_cost_tracking})"
        )


# Global configuration instance
_config: Optional[ObservabilityConfig] = None


def get_config() -> ObservabilityConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = ObservabilityConfig()
    return _config


def reset_config():
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
