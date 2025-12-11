"""
Central configuration manager for Phoenix Observability.

Loads configuration from environment variables (typically from .env file).
"""

import os
import threading
from typing import Optional, Dict

# Load environment variables from .env file
# Use try/except to handle cases where dotenv is not available or .env doesn't exist
try:
    from dotenv import load_dotenv
    load_dotenv()
except (ImportError, Exception):
    # If dotenv is not available or fails to load, continue without it
    # Environment variables can still be set via system environment
    pass


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
        self.max_queue_size: int = int(
            os.getenv("MAX_QUEUE_SIZE", "2048")
        )

        # Sanitization settings
        self.max_prompt_length: int = int(os.getenv("MAX_PROMPT_LENGTH", "10000"))
        self.max_response_length: int = int(
            os.getenv("MAX_RESPONSE_LENGTH", "50000")
        )
        self.max_context_length: int = int(
            os.getenv("MAX_CONTEXT_LENGTH", "50000")
        )
        self.max_value_length: int = int(
            os.getenv("MAX_VALUE_LENGTH", "1000")
        )
        self.max_span_attribute_length: int = int(
            os.getenv("MAX_SPAN_ATTRIBUTE_LENGTH", "1000")
        )
        
        # Rate limiting settings for external API calls
        self.rate_limit_requests_per_minute: int = int(
            os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60")
        )
        self.rate_limit_requests_per_second: int = int(
            os.getenv("RATE_LIMIT_REQUESTS_PER_SECOND", "10")
        )
        self.rate_limit_enabled: bool = os.getenv(
            "RATE_LIMIT_ENABLED", "true"
        ).lower() == "true"
        
        # HTTP connection pooling settings
        self.http_pool_connections: int = int(
            os.getenv("HTTP_POOL_CONNECTIONS", "10")
        )
        self.http_pool_maxsize: int = int(
            os.getenv("HTTP_POOL_MAXSIZE", "20")
        )
        self.http_timeout: int = int(
            os.getenv("HTTP_TIMEOUT", "30")
        )

        # OTLP exporter settings
        # SECURITY: Default to secure (False) - set OTLP_INSECURE=true only for local development
        # In production, always use secure connections (HTTPS/gRPC with TLS)
        self.otlp_insecure: bool = os.getenv(
            "OTLP_INSECURE", "false"
        ).lower() == "true"

        # Toxicity detection settings
        self.toxicity_detection_method: str = os.getenv(
            "TOXICITY_DETECTION_METHOD", "auto"
        ).lower()  # Options: 'auto', 'openai', 'perspective', 'heuristic'
        
        # Jailbreak detection and notification settings
        self.enable_jailbreak_warnings: bool = os.getenv(
            "ENABLE_JAILBREAK_WARNINGS", "true"
        ).lower() == "true"

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


# Global configuration instance with thread-safety
_config: Optional[ObservabilityConfig] = None
_config_lock = threading.Lock()


def get_config() -> ObservabilityConfig:
    """
    Get or create the global configuration instance (thread-safe).

    Returns:
        ObservabilityConfig instance.
    """
    global _config
    # Double-checked locking pattern for thread-safety
    if _config is None:
        with _config_lock:
            # Check again inside the lock to avoid race conditions
            if _config is None:
                _config = ObservabilityConfig()
    return _config


def reset_config():
    """
    Reset the global configuration (useful for testing, thread-safe).

    Note: This should only be used in tests. In production, the config
    should remain stable throughout the application lifecycle.
    """
    global _config
    with _config_lock:
        _config = None
