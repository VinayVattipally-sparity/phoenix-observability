"""
Structured logging utilities.

Provides JSON-formatted logging for better observability and log aggregation.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Formats log records as JSON with consistent structure for easy parsing
    by log aggregation systems (ELK, Splunk, Datadog, etc.).
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format.
            
        Returns:
            JSON-formatted log string.
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName or "<module>",
            "line": record.lineno,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
            }
        
        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add any additional attributes
        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName", "relativeCreated",
                "thread", "threadName", "exc_info", "exc_text", "stack_info",
                "extra_fields",
            ]:
                if not key.startswith("_"):
                    try:
                        json.dumps(value)  # Test if value is JSON serializable
                        log_data[key] = value
                    except (TypeError, ValueError):
                        log_data[key] = str(value)
        
        return json.dumps(log_data, default=str)


def setup_structured_logging(
    level: int = logging.INFO,
    handler: Optional[logging.Handler] = None,
    formatter: Optional[logging.Formatter] = None,
) -> None:
    """
    Set up structured JSON logging for the application.
    
    Args:
        level: Logging level (default: INFO).
        handler: Custom log handler (default: StreamHandler to stdout).
        formatter: Custom formatter (default: StructuredFormatter).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # Create handler if not provided
    if handler is None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
    
    # Create formatter if not provided
    if formatter is None:
        formatter = StructuredFormatter()
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_structured_logger(name: str) -> logging.Logger:
    """
    Get a logger configured for structured logging.
    
    Args:
        name: Logger name (typically __name__).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Add extra_fields method to logger for structured data
    def log_with_fields(level: int, msg: str, **kwargs: Any) -> None:
        """Log with additional structured fields."""
        extra = {"extra_fields": kwargs}
        logger.log(level, msg, extra=extra)
    
    # Add convenience methods
    logger.info_fields = lambda msg, **kwargs: log_with_fields(logging.INFO, msg, **kwargs)
    logger.warning_fields = lambda msg, **kwargs: log_with_fields(logging.WARNING, msg, **kwargs)
    logger.error_fields = lambda msg, **kwargs: log_with_fields(logging.ERROR, msg, **kwargs)
    logger.debug_fields = lambda msg, **kwargs: log_with_fields(logging.DEBUG, msg, **kwargs)
    
    return logger

