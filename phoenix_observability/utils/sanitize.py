"""
Sanitization utilities.

Prevents huge prompts/responses from being logged to Phoenix.
"""

import logging
from typing import Any, Dict, Optional

from phoenix_observability.config import get_config

logger = logging.getLogger(__name__)

TRUNCATION_MESSAGE = "... [truncated]"


def sanitize_prompt(prompt: Any, max_length: Optional[int] = None) -> str:
    """
    Sanitize and truncate prompt if too long.

    Args:
        prompt: Prompt to sanitize (can be str, list, dict, etc.). If None, returns empty string.
        max_length: Maximum length (defaults to config value). Must be positive if provided.

    Returns:
        Sanitized prompt string
    """
    # Handle None input
    if prompt is None:
        return ""
    
    config = get_config()
    max_len = max_length or config.max_prompt_length
    
    # Validate max_length if provided
    if max_length is not None and max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")

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
        response: Response to sanitize. If None, returns empty string.
        max_length: Maximum length (defaults to config value). Must be positive if provided.

    Returns:
        Sanitized response string
    """
    # Handle None input
    if response is None:
        return ""
    
    config = get_config()
    max_len = max_length or config.max_response_length
    
    # Validate max_length if provided
    if max_length is not None and max_length <= 0:
        raise ValueError(f"max_length must be positive, got {max_length}")

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


def sanitize_dict(
    data: Dict[str, Any], 
    max_value_length: Optional[int] = None
) -> Dict[str, Any]:
    """
    Sanitize dictionary values by truncating long strings.

    Args:
        data: Dictionary to sanitize. Must be a dict, not None.
        max_value_length: Maximum length for string values. Must be positive.

    Returns:
        Sanitized dictionary
        
    Raises:
        TypeError: If data is not a dict
        ValueError: If max_value_length is not positive
    """
    if not isinstance(data, dict):
        raise TypeError(f"data must be a dict, got {type(data).__name__}")
    
    config = get_config()
    max_len = max_value_length or config.max_value_length
    
    if max_len <= 0:
        raise ValueError(f"max_value_length must be positive, got {max_len}")
    
    sanitized = {}
    for key, value in data.items():
        if isinstance(value, str) and len(value) > max_len:
            sanitized[key] = value[:max_len] + TRUNCATION_MESSAGE
        elif isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, max_len)
        elif isinstance(value, list):
            sanitized[key] = [
                (
                    item[:max_len] + TRUNCATION_MESSAGE
                    if isinstance(item, str) and len(item) > max_len
                    else item
                )
                for item in value
            ]
        else:
            sanitized[key] = value
    return sanitized

