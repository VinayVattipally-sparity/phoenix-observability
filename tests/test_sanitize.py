"""
Unit tests for sanitize.py module.
"""

import pytest
from phoenix_observability.utils.sanitize import (
    sanitize_prompt,
    sanitize_response,
    sanitize_dict,
    TRUNCATION_MESSAGE,
)


class TestSanitizePrompt:
    """Tests for sanitize_prompt function."""

    def test_sanitize_prompt_string(self):
        """Test sanitization of string prompt."""
        prompt = "This is a test prompt"
        result = sanitize_prompt(prompt)
        assert result == prompt

    def test_sanitize_prompt_long(self):
        """Test truncation of long prompt."""
        long_prompt = "a" * 20000
        result = sanitize_prompt(long_prompt, max_length=100)
        assert len(result) <= 100
        assert TRUNCATION_MESSAGE in result

    def test_sanitize_prompt_none(self):
        """Test sanitization of None prompt."""
        result = sanitize_prompt(None)
        assert result == ""

    def test_sanitize_prompt_list(self):
        """Test sanitization of list prompt."""
        prompt = ["message1", "message2"]
        result = sanitize_prompt(prompt)
        assert isinstance(result, str)
        assert "message1" in result

    def test_sanitize_prompt_dict(self):
        """Test sanitization of dict prompt."""
        prompt = {"role": "user", "content": "Hello"}
        result = sanitize_prompt(prompt)
        assert isinstance(result, str)
        assert "Hello" in result

    def test_sanitize_prompt_custom_max_length(self):
        """Test sanitization with custom max length."""
        prompt = "a" * 100
        result = sanitize_prompt(prompt, max_length=50)
        assert len(result) <= 50

    def test_sanitize_prompt_invalid_max_length(self):
        """Test sanitization with invalid max length."""
        with pytest.raises(ValueError, match="must be positive"):
            sanitize_prompt("test", max_length=0)

    def test_sanitize_prompt_negative_max_length(self):
        """Test sanitization with negative max length."""
        with pytest.raises(ValueError, match="must be positive"):
            sanitize_prompt("test", max_length=-1)


class TestSanitizeResponse:
    """Tests for sanitize_response function."""

    def test_sanitize_response_string(self):
        """Test sanitization of string response."""
        response = "This is a test response"
        result = sanitize_response(response)
        assert result == response

    def test_sanitize_response_long(self):
        """Test truncation of long response."""
        long_response = "a" * 60000
        result = sanitize_response(long_response, max_length=100)
        assert len(result) <= 100
        assert TRUNCATION_MESSAGE in result

    def test_sanitize_response_none(self):
        """Test sanitization of None response."""
        result = sanitize_response(None)
        assert result == ""

    def test_sanitize_response_list(self):
        """Test sanitization of list response."""
        response = ["item1", "item2"]
        result = sanitize_response(response)
        assert isinstance(result, str)

    def test_sanitize_response_dict(self):
        """Test sanitization of dict response."""
        response = {"result": "success", "data": "value"}
        result = sanitize_response(response)
        assert isinstance(result, str)

    def test_sanitize_response_invalid_max_length(self):
        """Test sanitization with invalid max length."""
        with pytest.raises(ValueError, match="must be positive"):
            sanitize_response("test", max_length=0)


class TestSanitizeDict:
    """Tests for sanitize_dict function."""

    def test_sanitize_dict_simple(self):
        """Test sanitization of simple dictionary."""
        data = {"key1": "value1", "key2": "value2"}
        result = sanitize_dict(data)
        assert result == data

    def test_sanitize_dict_long_value(self):
        """Test truncation of long dictionary values."""
        data = {"key": "a" * 2000}
        result = sanitize_dict(data, max_value_length=100)
        assert len(result["key"]) <= 100 + len(TRUNCATION_MESSAGE)
        assert TRUNCATION_MESSAGE in result["key"]

    def test_sanitize_dict_nested(self):
        """Test sanitization of nested dictionary."""
        data = {
            "level1": {
                "level2": {
                    "key": "a" * 2000
                }
            }
        }
        result = sanitize_dict(data, max_value_length=100)
        assert TRUNCATION_MESSAGE in result["level1"]["level2"]["key"]

    def test_sanitize_dict_with_list(self):
        """Test sanitization of dictionary with list."""
        data = {
            "items": ["short", "a" * 2000, "normal"]
        }
        result = sanitize_dict(data, max_value_length=100)
        assert len(result["items"][1]) <= 100 + len(TRUNCATION_MESSAGE)

    def test_sanitize_dict_not_dict(self):
        """Test sanitization with non-dict input."""
        with pytest.raises(TypeError, match="must be a dict"):
            sanitize_dict("not a dict")

    def test_sanitize_dict_invalid_max_length(self):
        """Test sanitization with invalid max length."""
        with pytest.raises(ValueError, match="must be positive"):
            sanitize_dict({"key": "value"}, max_value_length=0)

    def test_sanitize_dict_negative_max_length(self):
        """Test sanitization with negative max length."""
        with pytest.raises(ValueError, match="must be positive"):
            sanitize_dict({"key": "value"}, max_value_length=-1)

    def test_sanitize_dict_preserves_non_strings(self):
        """Test that non-string values are preserved."""
        data = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
        }
        result = sanitize_dict(data)
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True
        assert result["none"] is None

