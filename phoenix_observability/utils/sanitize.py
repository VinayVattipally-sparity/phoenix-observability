"""
Sanitization utilities.

Prevents huge prompts/responses from being logged to Phoenix.
"""

import logging
from typing import Any, Optional

from phoenix_observability.config import get_config

logger = logging.getLogger(__name__)

TRUNCATION_MESSAGE = "... [truncated]"


def sanitize_prompt(prompt: Any, max_length: Optional[int] = None) -> str:
    """
    Sanitize and truncate prompt if too long.

    Args:
        prompt: Prompt to sanitize (can be str, list, dict, etc.)
        max_length: Maximum length (defaults to config value)

    Returns:
        Sanitized prompt string
    """
    config = get_config()
    max_len = max_length or config.max_prompt_length

    # Convert to string
    if isinstance(prompt, str):
        prompt_str = prompt
    elif isinstance(prompt, (list, dict)):
        prompt_str = str(prompt)
    else:
        prompt_str = str(prompt)

    # Truncate if needed
    if len(prompt_str) > max_len:
        truncated = prompt_str[: max_len - len(TRUNCATION_MESSAGE)]
        return truncated + TRUNCATION_MESSAGE

    return prompt_str


def sanitize_response(response: Any, max_length: Optional[int] = None) -> str:
    """
    Sanitize and truncate response if too long.

    Args:
        response: Response to sanitize
        max_length: Maximum length (defaults to config value)

    Returns:
        Sanitized response string
    """
    config = get_config()
    max_len = max_length or config.max_response_length

    # Convert to string
    if isinstance(response, str):
        response_str = response
    elif isinstance(response, (list, dict)):
        response_str = str(response)
    else:
        response_str = str(response)

    # Truncate if needed
    if len(response_str) > max_len:
        truncated = response_str[: max_len - len(TRUNCATION_MESSAGE)]
        return truncated + TRUNCATION_MESSAGE

    return response_str


def sanitize_dict(data: dict, max_value_length: int = 1000) -> dict:
    """
    Sanitize dictionary values by truncating long strings.

    Args:
        data: Dictionary to sanitize
        max_value_length: Maximum length for string values

    Returns:
        Sanitized dictionary
    """
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str) and len(value) > max_value_length:
            sanitized[key] = value[:max_value_length] + TRUNCATION_MESSAGE
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, max_value_length)
        elif isinstance(value, list):
            sanitized[key] = [
                (
                    item[:max_value_length] + TRUNCATION_MESSAGE
                    if isinstance(item, str) and len(item) > max_value_length
                    else item
                )
                for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized

