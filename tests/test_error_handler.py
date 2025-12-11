"""
Unit tests for error_handler.py module.
"""

import pytest
from unittest.mock import Mock
from phoenix_observability.instrumentation.error_handler import (
    attach_exception_to_span,
    handle_error,
)


class TestAttachExceptionToSpan:
    """Tests for attach_exception_to_span function."""

    def test_attach_exception_with_stack_trace(self, sample_span):
        """Test attaching exception with stack trace."""
        exception = ValueError("Test error message")
        attach_exception_to_span(sample_span, exception, include_stack_trace=True)
        
        # Check that exception was recorded
        assert sample_span.record_exception.called
        
        # Check that status was set
        assert sample_span.set_status.called
        
        # Check that attributes were set
        assert sample_span.set_attribute.called
        calls = {call[0][0]: call[0][1] for call in sample_span.set_attribute.call_args_list}
        assert "error.type" in calls
        assert calls["error.type"] == "ValueError"
        assert "error.message" in calls
        assert "error.is_failure" in calls
        assert calls["error.is_failure"] is True
        assert "error.traceback" in calls

    def test_attach_exception_without_stack_trace(self, sample_span):
        """Test attaching exception without stack trace."""
        exception = ValueError("Test error message")
        attach_exception_to_span(sample_span, exception, include_stack_trace=False)
        
        # Check that exception was recorded
        assert sample_span.record_exception.called
        
        # Check that traceback was not set
        calls = {call[0][0]: call[0][1] for call in sample_span.set_attribute.call_args_list}
        assert "error.traceback" not in calls

    def test_attach_exception_sanitizes_message(self, sample_span):
        """Test that error messages are sanitized."""
        exception = ValueError("Error with key sk-abc123")
        attach_exception_to_span(sample_span, exception)
        
        calls = {call[0][0]: call[0][1] for call in sample_span.set_attribute.call_args_list}
        error_message = calls.get("error.message", "")
        # Should not contain the API key
        assert "sk-abc123" not in error_message


class TestHandleError:
    """Tests for handle_error function."""

    def test_handle_error_basic(self, sample_span):
        """Test basic error handling."""
        exception = ValueError("Test error")
        handle_error(sample_span, exception)
        
        # Should call attach_exception_to_span
        assert sample_span.record_exception.called

    def test_handle_error_with_context(self, sample_span):
        """Test error handling with context."""
        exception = ValueError("Test error")
        context = {"key1": "value1", "key2": "value2"}
        handle_error(sample_span, exception, context=context)
        
        # Check that context was added
        assert sample_span.set_attribute.called
        calls = {call[0][0]: call[0][1] for call in sample_span.set_attribute.call_args_list}
        assert "error.context.key1" in calls
        assert calls["error.context.key1"] == "value1"
        assert "error.context.key2" in calls
        assert calls["error.context.key2"] == "value2"

    def test_handle_error_sanitizes_context(self, sample_span):
        """Test that context values are sanitized."""
        exception = ValueError("Test error")
        context = {"api_key": "sk-abc123", "token": "secret123"}
        handle_error(sample_span, exception, context=context)
        
        calls = {call[0][0]: call[0][1] for call in sample_span.set_attribute.call_args_list}
        # Sensitive data should be redacted
        api_key_value = calls.get("error.context.api_key", "")
        assert "sk-abc123" not in api_key_value

