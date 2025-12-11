"""
Unit tests for structured logging.
"""

import json
import logging
import sys
from io import StringIO
from phoenix_observability.logging.structured import (
    StructuredFormatter,
    setup_structured_logging,
    get_structured_logger,
)


class TestStructuredFormatter:
    """Tests for StructuredFormatter class."""

    def test_format_basic(self):
        """Test basic log formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test"
        assert "timestamp" in data
        assert data["module"] == "test"
        assert data["function"] == "<module>"
        assert data["line"] == 10

    def test_format_with_exception(self):
        """Test formatting with exception."""
        formatter = StructuredFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info(),
            )
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["level"] == "ERROR"
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert "Test error" in data["exception"]["message"]

    def test_format_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.extra_fields = {"user_id": "123", "request_id": "abc"}
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["user_id"] == "123"
        assert data["request_id"] == "abc"


class TestSetupStructuredLogging:
    """Tests for setup_structured_logging function."""

    def test_setup_default(self):
        """Test setting up structured logging with defaults."""
        # Capture output
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        
        setup_structured_logging(level=logging.INFO, handler=handler)
        
        logger = logging.getLogger("test")
        logger.info("Test message")
        
        output = stream.getvalue()
        data = json.loads(output.strip())
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"


class TestGetStructuredLogger:
    """Tests for get_structured_logger function."""

    def test_get_structured_logger(self):
        """Test getting structured logger."""
        logger = get_structured_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_logger_with_fields(self):
        """Test logger with extra fields."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        
        logger = get_structured_logger("test")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        logger.info_fields("Test message", user_id="123", action="test")
        
        output = stream.getvalue()
        data = json.loads(output.strip())
        assert data["user_id"] == "123"
        assert data["action"] == "test"

