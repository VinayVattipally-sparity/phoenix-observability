"""
Central configuration manager for Phoenix Observability.

Loads configuration from environment variables (typically from .env file).
"""

import os
from typing import Optional, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ObservabilityConfig:
    """Central configuration for observability settings."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Phoenix endpoint - with fallback to default
        # First try project's .env file, then fallback to default
        self.phoenix_endpoint: str = os.getenv(
            "PHOENIX_ENDPOINT", "https://sparity-phoenix.com"
        )

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

        # Toxicity detection settings
        self.toxicity_detection_method: str = os.getenv(
            "TOXICITY_DETECTION_METHOD", "auto"
        ).lower()  # Options: 'auto', 'openai', 'perspective', 'heuristic'

        # Authentication settings (with fallback to defaults)
        # OTLP authentication headers (for Phoenix/OTLP endpoint authentication)
        # Format: "header1=value1,header2=value2" or "Authorization=Bearer token"
        self.otlp_headers: Optional[str] = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
        
        # Default authentication (used if not in project's .env)
        # These are package defaults - projects can override via .env
        self.default_otlp_headers: Optional[str] = os.getenv(
            "PHOENIX_OTLP_DEFAULT_HEADERS"  # Package-level default (rarely set)
        )
        
        # Phoenix API authentication (if Phoenix requires API key)
        self.phoenix_api_key: Optional[str] = os.getenv("PHOENIX_API_KEY")
        self.default_phoenix_api_key: Optional[str] = os.getenv(
            "PHOENIX_DEFAULT_API_KEY"  # Package-level default (rarely set)
        )
        
        # OTLP authentication token (alternative to headers)
        self.otlp_token: Optional[str] = os.getenv("OTLP_TOKEN")
        self.default_otlp_token: Optional[str] = os.getenv(
            "OTLP_DEFAULT_TOKEN"  # Package-level default (rarely set)
        )

    def get_otlp_headers(self) -> Optional[Dict[str, str]]:
        """
        Get OTLP authentication headers with fallback to defaults.
        
        Priority:
        1. OTEL_EXPORTER_OTLP_HEADERS from project's .env
        2. PHOENIX_OTLP_DEFAULT_HEADERS (package default)
        3. Constructed from OTLP_TOKEN or PHOENIX_API_KEY if available
        4. None (no authentication)
        
        Returns:
            Dictionary of headers or None if no authentication configured
        """
        # Try project's .env first
        headers_str = self.otlp_headers
        
        # Fallback to package default
        if not headers_str:
            headers_str = self.default_otlp_headers
        
        # If still no headers, try constructing from token/API key
        if not headers_str:
            # Try OTLP_TOKEN first (project's .env)
            token = self.otlp_token
            if not token:
                # Fallback to default token
                token = self.default_otlp_token
            
            # If token found, construct Authorization header
            if token:
                headers_str = f"Authorization=Bearer {token}"
            else:
                # Try Phoenix API key
                api_key = self.phoenix_api_key
                if not api_key:
                    api_key = self.default_phoenix_api_key
                
                if api_key:
                    headers_str = f"Authorization=Bearer {api_key}"
        
        # Parse headers string into dictionary
        if headers_str:
            headers = {}
            for header_pair in headers_str.split(","):
                if "=" in header_pair:
                    key, value = header_pair.split("=", 1)
                    headers[key.strip().lower()] = value.strip()
            return headers if headers else None
        
        return None

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
