"""
Cost tracking utilities.

Converts token usage to cost based on model pricing.
"""

import functools
import logging
from typing import Dict, Optional, Any, Tuple

from opentelemetry import trace

from phoenix_observability.utils.metrics import (
    record_cost_calculated,
    record_cost_amount,
)

logger = logging.getLogger(__name__)

# Model pricing per 1M tokens (input/output)
# Prices in USD, update as needed
# Sources: Official pricing pages (as of 2024)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI Models
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4o-2024-08-06": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.6},
    # GPT-4.1 Models (pricing as of 2025-04-14)
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-2025-04-14": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
    "gpt-4.1-mini-2025-04-14": {"input": 0.4, "output": 1.6},
    "gpt-4.1-nano": {"input": 0.1, "output": 0.4},
    "gpt-4.1-nano-2025-04-14": {"input": 0.1, "output": 0.4},
    
    # Anthropic Claude Models
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku": {"input": 1.0, "output": 5.0},
    "claude-3-5-haiku-20241022": {"input": 1.0, "output": 5.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    
    # Google Gemini Models
    "gemini-pro": {"input": 0.5, "output": 1.5},
    "gemini-pro-vision": {"input": 0.5, "output": 1.5},
    "gemini-ultra": {"input": 1.25, "output": 5.0},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    "gemini-1.5-pro-latest": {"input": 1.25, "output": 5.0},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-latest": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash-thinking-exp": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash-thinking-exp-001": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-exp": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-thinking-exp": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-thinking-exp-001": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-thinking-exp-002": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-thinking-exp-003": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-thinking-exp-004": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-thinking-exp-005": {"input": 0.075, "output": 0.30},
}

# Initialize sorted keys after MODEL_PRICING is defined
_SORTED_PRICING_KEYS = sorted(MODEL_PRICING.keys(), key=len, reverse=True)


@functools.lru_cache(maxsize=128)
def _find_pricing_for_model(model_name: str) -> Optional[Dict[str, float]]:
    """
    Find pricing dictionary for a model (cached).
    
    Args:
        model_name: Name of the model (lowercased).
        
    Returns:
        Pricing dictionary or None if not found.
    """
    # First, try exact match
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    
    # Try partial matches
    for key in _SORTED_PRICING_KEYS:
        key_lower = key.lower()
        if not (key_lower in model_name or model_name in key_lower):
            continue
        
        # Additional check: ensure we're matching the right model family
        key_parts = key_lower.split("-")
        model_parts = model_name.split("-")
        
        # Match if major parts align (e.g., "gemini", "2.0", "flash")
        if len(key_parts) >= 2:
            if key_parts[0] in model_parts and key_parts[1] in model_parts:
                return MODEL_PRICING[key]
        else:
            # For simple names, just check containment
            return MODEL_PRICING[key]
    
    return None


def calculate_cost(
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    custom_pricing: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate cost based on token usage.

    Args:
        model_name: Name of the model. Must be a non-empty string.
        input_tokens: Number of input tokens. Must be non-negative.
        output_tokens: Number of output tokens. Must be non-negative.
        custom_pricing: Optional custom pricing dict with 'input' and 'output' keys

    Returns:
        Total cost in USD
        
    Raises:
        TypeError: If model_name is not a string
        ValueError: If model_name is empty, or token counts are negative
    """
    # Validate model_name
    if not isinstance(model_name, str):
        raise TypeError(f"model_name must be a string, got {type(model_name).__name__}")
    if not model_name.strip():
        raise ValueError("model_name cannot be empty")
    
    # Validate token counts
    if not isinstance(input_tokens, int):
        raise TypeError(f"input_tokens must be an integer, got {type(input_tokens).__name__}")
    if input_tokens < 0:
        raise ValueError(f"input_tokens must be non-negative, got {input_tokens}")
    
    if not isinstance(output_tokens, int):
        raise TypeError(f"output_tokens must be an integer, got {type(output_tokens).__name__}")
    if output_tokens < 0:
        raise ValueError(f"output_tokens must be non-negative, got {output_tokens}")
    
    # Validate custom_pricing if provided
    if custom_pricing is not None:
        if not isinstance(custom_pricing, dict):
            raise TypeError(f"custom_pricing must be a dict, got {type(custom_pricing).__name__}")
        if "input" not in custom_pricing or "output" not in custom_pricing:
            raise ValueError("custom_pricing must contain 'input' and 'output' keys")
        if not isinstance(custom_pricing["input"], (int, float)) or custom_pricing["input"] < 0:
            raise ValueError("custom_pricing['input'] must be a non-negative number")
        if not isinstance(custom_pricing["output"], (int, float)) or custom_pricing["output"] < 0:
            raise ValueError("custom_pricing['output'] must be a non-negative number")
    
    # Use custom pricing if provided
    if custom_pricing:
        pricing = custom_pricing
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        # Record metrics
        record_cost_calculated(model_name)
        if total_cost > 0:
            record_cost_amount(total_cost, model_name)
        return total_cost

    # Try to find model pricing using cached lookup
    model_lower = model_name.lower()
    pricing = _find_pricing_for_model(model_lower)
    
    if pricing:
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        # Record metrics
        record_cost_calculated(model_name)
        if total_cost > 0:
            record_cost_amount(total_cost, model_name)
        return total_cost

    # Fallback: infer provider and use provider-specific default
    if "gpt" in model_lower or "openai" in model_lower:
        provider_default = MODEL_PRICING["gpt-3.5-turbo"]
        logger.warning(
            f"Unknown OpenAI model pricing for {model_name}, using default GPT-3.5 pricing"
        )
    elif "gemini" in model_lower or "google" in model_lower:
        provider_default = MODEL_PRICING["gemini-1.5-flash"]
        logger.warning(
            f"Unknown Gemini model pricing for {model_name}, using default Gemini 1.5 Flash pricing"
        )
    elif "claude" in model_lower or "anthropic" in model_lower:
        provider_default = MODEL_PRICING["claude-3-haiku"]
        logger.warning(
            f"Unknown Claude model pricing for {model_name}, using default Claude 3 Haiku pricing"
        )
    else:
        provider_default = MODEL_PRICING["gpt-3.5-turbo"]
        logger.warning(
            f"Unknown model pricing for {model_name}, using default GPT-3.5 pricing"
        )
    
    pricing = provider_default
    
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    # Record metrics
    record_cost_calculated(model_name)
    if total_cost > 0:
        record_cost_amount(total_cost, model_name)
    return total_cost


def attach_cost_to_span(
    span: trace.Span,
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    total_tokens: Optional[int] = None,
    custom_pricing: Optional[Dict[str, float]] = None,
) -> None:
    """
    Attach cost information to a span.

    Args:
        span: OpenTelemetry span. Must not be None.
        model_name: Name of the model. Must be a non-empty string.
        input_tokens: Number of input tokens. Must be non-negative.
        output_tokens: Number of output tokens. Must be non-negative.
        total_tokens: Total tokens (if provided, used for validation). Must be non-negative.
        custom_pricing: Optional custom pricing
        
    Raises:
        TypeError: If span is None or invalid types provided
        ValueError: If model_name is empty or token counts are negative
    """
    # Validate span
    if span is None:
        raise TypeError("span cannot be None")
    
    # Validate model_name
    if not isinstance(model_name, str):
        raise TypeError(f"model_name must be a string, got {type(model_name).__name__}")
    if not model_name.strip():
        raise ValueError("model_name cannot be empty")
    
    # Validate token counts
    if not isinstance(input_tokens, int):
        raise TypeError(f"input_tokens must be an integer, got {type(input_tokens).__name__}")
    if input_tokens < 0:
        raise ValueError(f"input_tokens must be non-negative, got {input_tokens}")
    
    if not isinstance(output_tokens, int):
        raise TypeError(f"output_tokens must be an integer, got {type(output_tokens).__name__}")
    if output_tokens < 0:
        raise ValueError(f"output_tokens must be non-negative, got {output_tokens}")
    
    if total_tokens is not None:
        if not isinstance(total_tokens, int):
            raise TypeError(f"total_tokens must be an integer, got {type(total_tokens).__name__}")
        if total_tokens < 0:
            raise ValueError(f"total_tokens must be non-negative, got {total_tokens}")
    
    if total_tokens and (input_tokens + output_tokens) != total_tokens:
        logger.warning(
            f"Token mismatch: input={input_tokens}, output={output_tokens}, "
            f"total={total_tokens}"
        )

    cost = calculate_cost(model_name, input_tokens, output_tokens, custom_pricing)

    # Set cost
    span.set_attribute("llm.cost_usd", cost)
    
    # Set token counts in nested structure (matching previous implementation)
    span.set_attribute("llm.token_count.prompt", input_tokens)
    span.set_attribute("llm.token_count.completion", output_tokens)
    span.set_attribute("llm.token_count.total", input_tokens + output_tokens)
    
    # Also set flat structure for compatibility
    span.set_attribute("llm.tokens.prompt", input_tokens)
    span.set_attribute("llm.tokens.completion", output_tokens)
    span.set_attribute("llm.tokens.total", input_tokens + output_tokens)
    
    # Keep old names for backward compatibility
    span.set_attribute("llm.cost.usd", cost)
    span.set_attribute("llm.tokens.input", input_tokens)
    span.set_attribute("llm.tokens.output", output_tokens)
