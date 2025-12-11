"""
Unit tests for security.py module.
"""

import pytest
from phoenix_observability.utils.security import (
    validate_api_key_format,
    sanitize_url,
    sanitize_name,
    redact_sensitive_data,
    sanitize_error_message,
    sanitize_stack_trace,
)


class TestValidateAPIKeyFormat:
    """Tests for API key format validation."""

    def test_openai_key_valid(self):
        """Test valid OpenAI API key."""
        key = "sk-" + "a" * 32
        assert validate_api_key_format(key, "openai") is True

    def test_openai_key_invalid_short(self):
        """Test invalid OpenAI API key (too short)."""
        key = "sk-abc"
        assert validate_api_key_format(key, "openai") is False

    def test_openai_key_invalid_prefix(self):
        """Test invalid OpenAI API key (wrong prefix)."""
        key = "pk-" + "a" * 32
        assert validate_api_key_format(key, "openai") is False

    def test_anthropic_key_valid(self):
        """Test valid Anthropic API key."""
        key = "sk-ant-" + "a" * 95
        assert validate_api_key_format(key, "anthropic") is True

    def test_anthropic_key_invalid(self):
        """Test invalid Anthropic API key."""
        key = "sk-ant-abc"
        assert validate_api_key_format(key, "anthropic") is False

    def test_gemini_key_valid(self):
        """Test valid Gemini API key."""
        key = "a" * 20
        assert validate_api_key_format(key, "gemini") is True

    def test_gemini_key_invalid(self):
        """Test invalid Gemini API key."""
        key = "abc"
        assert validate_api_key_format(key, "gemini") is False

    def test_none_key(self):
        """Test None key."""
        assert validate_api_key_format(None, "openai") is False

    def test_empty_key(self):
        """Test empty key."""
        assert validate_api_key_format("", "openai") is False


class TestSanitizeURL:
    """Tests for URL sanitization."""

    def test_valid_https_url(self):
        """Test valid HTTPS URL."""
        url = "https://example.com"
        assert sanitize_url(url) == "https://example.com"

    def test_valid_http_url(self):
        """Test valid HTTP URL."""
        url = "http://localhost:6006"
        assert sanitize_url(url) == "http://localhost:6006"

    def test_url_with_path(self):
        """Test URL with path."""
        url = "https://example.com/v1/traces"
        assert sanitize_url(url) == "https://example.com/v1/traces"

    def test_url_strips_query_params(self):
        """Test that query parameters are stripped."""
        url = "https://example.com/path?key=value&token=secret123"
        result = sanitize_url(url)
        assert "key=value" not in result
        assert "token=secret123" not in result

    def test_url_missing_scheme(self):
        """Test URL without scheme raises error."""
        with pytest.raises(ValueError, match="scheme"):
            sanitize_url("example.com")

    def test_url_missing_hostname(self):
        """Test URL without hostname raises error."""
        with pytest.raises(ValueError, match="hostname"):
            sanitize_url("https://")

    def test_url_invalid_scheme(self):
        """Test URL with invalid scheme raises error."""
        with pytest.raises(ValueError):
            sanitize_url("javascript:alert('xss')")

    def test_url_none(self):
        """Test None URL raises error."""
        with pytest.raises(ValueError):
            sanitize_url(None)

    def test_url_empty(self):
        """Test empty URL raises error."""
        with pytest.raises(ValueError):
            sanitize_url("")


class TestSanitizeName:
    """Tests for name sanitization."""

    def test_valid_name(self):
        """Test valid name."""
        name = "my-service-v1"
        assert sanitize_name(name) == "my-service-v1"

    def test_name_with_underscore(self):
        """Test name with underscore."""
        name = "my_service"
        assert sanitize_name(name) == "my_service"

    def test_name_with_dot(self):
        """Test name with dot."""
        name = "my.service"
        assert sanitize_name(name) == "my.service"

    def test_name_too_long(self):
        """Test name exceeding max length."""
        name = "a" * 101
        with pytest.raises(ValueError, match="exceeds maximum length"):
            sanitize_name(name)

    def test_name_path_traversal(self):
        """Test name with path traversal pattern."""
        with pytest.raises(ValueError):
            sanitize_name("../../../etc/passwd")

    def test_name_html_injection(self):
        """Test name with HTML injection."""
        with pytest.raises(ValueError):
            sanitize_name("<script>alert('xss')</script>")

    def test_name_command_injection(self):
        """Test name with command injection."""
        with pytest.raises(ValueError):
            sanitize_name("name; rm -rf /")

    def test_name_none(self):
        """Test None name raises error."""
        with pytest.raises(ValueError):
            sanitize_name(None)

    def test_name_empty(self):
        """Test empty name raises error."""
        with pytest.raises(ValueError):
            sanitize_name("")


class TestRedactSensitiveData:
    """Tests for sensitive data redaction."""

    def test_redact_api_key(self):
        """Test redaction of API key."""
        text = "Error with key sk-abc123xyz456"
        result = redact_sensitive_data(text)
        assert "sk-abc123xyz456" not in result
        assert "[REDACTED]" in result

    def test_redact_bearer_token(self):
        """Test redaction of Bearer token."""
        text = "Authorization: Bearer token12345678901234567890"
        result = redact_sensitive_data(text)
        assert "token12345678901234567890" not in result
        assert "[REDACTED]" in result

    def test_redact_password(self):
        """Test redaction of password."""
        text = 'password="secret123"'
        result = redact_sensitive_data(text)
        assert "secret123" not in result
        assert "[REDACTED]" in result

    def test_redact_secret(self):
        """Test redaction of secret."""
        text = 'secret="my-secret-key-12345"'
        result = redact_sensitive_data(text)
        assert "my-secret-key-12345" not in result
        assert "[REDACTED]" in result

    def test_no_sensitive_data(self):
        """Test text without sensitive data."""
        text = "This is a normal error message"
        result = redact_sensitive_data(text)
        assert result == text

    def test_redact_none(self):
        """Test redaction of None."""
        result = redact_sensitive_data(None)
        assert result == ""

    def test_redact_empty(self):
        """Test redaction of empty string."""
        result = redact_sensitive_data("")
        assert result == ""


class TestSanitizeErrorMessage:
    """Tests for error message sanitization."""

    def test_sanitize_error_with_key(self):
        """Test sanitization of error with API key."""
        error_msg = "Failed with key sk-abc123"
        result = sanitize_error_message(error_msg)
        assert "sk-abc123" not in result
        assert "[REDACTED]" in result

    def test_sanitize_error_with_exception(self):
        """Test sanitization of error with exception."""
        error_msg = "Error occurred"
        exception = ValueError("Key: sk-abc123")
        result = sanitize_error_message(error_msg, exception)
        assert "sk-abc123" not in result

    def test_sanitize_error_none(self):
        """Test sanitization of None error."""
        result = sanitize_error_message(None)
        assert result == ""


class TestSanitizeStackTrace:
    """Tests for stack trace sanitization."""

    def test_sanitize_stack_trace_with_key(self):
        """Test sanitization of stack trace with API key."""
        trace = "File test.py, line 10\n  key = 'sk-abc123'\nValueError"
        result = sanitize_stack_trace(trace)
        assert "sk-abc123" not in result
        assert "[REDACTED]" in result

    def test_sanitize_stack_trace_empty(self):
        """Test sanitization of empty stack trace."""
        result = sanitize_stack_trace("")
        assert result == ""

    def test_sanitize_stack_trace_none(self):
        """Test sanitization of None stack trace."""
        result = sanitize_stack_trace(None)
        assert result == ""

