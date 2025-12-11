"""
Security utilities for input validation, sanitization, and sensitive data handling.

This module provides functions to validate API keys, sanitize user inputs,
and redact sensitive information from logs and error messages.
"""

import re
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# API key patterns for validation
OPENAI_API_KEY_PATTERN = re.compile(r'^sk-[a-zA-Z0-9]{32,}$')
ANTHROPIC_API_KEY_PATTERN = re.compile(r'^sk-ant-[a-zA-Z0-9\-_]{95,}$')
GEMINI_API_KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{20,}$')  # Gemini keys are alphanumeric, typically 39 chars
GOOGLE_API_KEY_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{20,}$')  # Google API keys are similar

# Patterns for detecting sensitive data
SENSITIVE_PATTERNS = [
    re.compile(r'sk-[a-zA-Z0-9]{10,}', re.IGNORECASE),  # API keys (OpenAI, Anthropic, etc.)
    re.compile(r'Bearer\s+[a-zA-Z0-9\-_]{20,}', re.IGNORECASE),  # Bearer tokens
    re.compile(r'api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-_]{10,}', re.IGNORECASE),
    re.compile(r'token["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-_]{10,}', re.IGNORECASE),
    re.compile(r'password["\']?\s*[:=]\s*["\']?[^\s"\']+', re.IGNORECASE),
    re.compile(r'secret["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-_]{10,}', re.IGNORECASE),
]

REDACTION_MESSAGE = "[REDACTED]"


def validate_api_key_format(key: str, key_type: str = "openai") -> bool:
    """
    Validate API key format based on provider.
    
    Args:
        key: API key to validate.
        key_type: Type of API key ('openai', 'anthropic', 'gemini', 'google').
        
    Returns:
        True if format is valid, False otherwise.
    """
    if not key or not isinstance(key, str):
        return False
    
    key = key.strip()
    
    if key_type.lower() == "openai":
        return bool(OPENAI_API_KEY_PATTERN.match(key))
    elif key_type.lower() == "anthropic":
        return bool(ANTHROPIC_API_KEY_PATTERN.match(key))
    elif key_type.lower() in ("gemini", "google"):
        return bool(GEMINI_API_KEY_PATTERN.match(key))
    
    # Default: check minimum length
    return len(key) >= 20


def sanitize_url(url: str) -> str:
    """
    Sanitize and validate URL format.
    
    Args:
        url: URL to sanitize.
        
    Returns:
        Sanitized URL.
        
    Raises:
        ValueError: If URL format is invalid.
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")
    
    url = url.strip()
    
    # Basic URL validation
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            raise ValueError("URL must include a scheme (http:// or https://)")
        if not parsed.netloc:
            raise ValueError("URL must include a hostname")
        
            # Only allow http and https schemes (check before netloc to get better error)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"URL scheme must be http or https, got {parsed.scheme}")
        
        # Remove any query parameters that might contain sensitive data
        # Keep only the base URL
        sanitized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        return sanitized.rstrip('/')
    except Exception as e:
        raise ValueError(f"Invalid URL format: {e}")


def sanitize_name(name: str, max_length: int = 100, allow_special: bool = False) -> str:
    """
    Sanitize project/service names to prevent injection.
    
    Args:
        name: Name to sanitize.
        max_length: Maximum allowed length.
        allow_special: Whether to allow special characters.
        
    Returns:
        Sanitized name.
        
    Raises:
        ValueError: If name is invalid.
    """
    if not name or not isinstance(name, str):
        raise ValueError("Name must be a non-empty string")
    
    name = name.strip()
    
    if len(name) > max_length:
        raise ValueError(f"Name exceeds maximum length of {max_length} characters")
    
    if not allow_special:
        # Only allow alphanumeric, hyphens, underscores, and dots
        if not re.match(r'^[a-zA-Z0-9._-]+$', name):
            raise ValueError(
                "Name can only contain alphanumeric characters, dots, hyphens, and underscores"
            )
    
    # Prevent common injection patterns
    dangerous_patterns = [
        r'\.\.',  # Path traversal
        r'[<>"\']',  # HTML/script injection
        r'[;&|`$]',  # Command injection
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, name):
            raise ValueError(f"Name contains invalid characters: {name}")
    
    return name


def redact_sensitive_data(text: str) -> str:
    """
    Redact sensitive information from text (API keys, tokens, etc.).
    
    Args:
        text: Text that may contain sensitive data.
        
    Returns:
        Text with sensitive data redacted.
    """
    if not text or not isinstance(text, str):
        return str(text) if text else ""
    
    redacted = text
    
    # Apply each sensitive pattern
    for pattern in SENSITIVE_PATTERNS:
        redacted = pattern.sub(REDACTION_MESSAGE, redacted)
    
    return redacted


def sanitize_error_message(error_msg: str, exception: Optional[Exception] = None) -> str:
    """
    Sanitize error message to remove sensitive data.
    
    Args:
        error_msg: Error message to sanitize.
        exception: Optional exception object.
        
    Returns:
        Sanitized error message.
    """
    if not error_msg:
        return ""
    
    # Redact sensitive patterns
    sanitized = redact_sensitive_data(str(error_msg))
    
    # If exception provided, also sanitize exception message
    if exception:
        exc_msg = str(exception)
        sanitized_exc = redact_sensitive_data(exc_msg)
        if sanitized_exc != exc_msg:
            sanitized = f"{sanitized} (Exception details redacted)"
    
    return sanitized


def sanitize_stack_trace(traceback_str: str) -> str:
    """
    Sanitize stack trace to remove sensitive data.
    
    Args:
        traceback_str: Stack trace string.
        
    Returns:
        Sanitized stack trace.
    """
    if not traceback_str:
        return ""
    
    # Split by lines and sanitize each line
    lines = traceback_str.split('\n')
    sanitized_lines = []
    
    for line in lines:
        sanitized_line = redact_sensitive_data(line)
        sanitized_lines.append(sanitized_line)
    
    return '\n'.join(sanitized_lines)

