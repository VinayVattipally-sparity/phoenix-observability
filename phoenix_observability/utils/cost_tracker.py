"""
Cost tracking utilities.

Converts token usage to cost based on model pricing.
"""

import logging
from typing import Dict, Optional, Any

from opentelemetry import trace

logger = logging.getLogger(__name__)

# Model pricing per 1M tokens (input/output)
# Prices in USD, update as needed
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "gemini-pro": {"input": 0.5, "output": 1.5},
    "gemini-ultra": {"input": 1.25, "output": 5.0},
    # Add more models as needed
}


def calculate_cost(
    model_name: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    custom_pricing: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate cost based on token usage.

    Args:
        model_name: Name of the model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        custom_pricing: Optional custom pricing dict with 'input' and 'output' keys

    Returns:
        Total cost in USD
    """
    if custom_pricing:
        pricing = custom_pricing
    else:
        # Try to find model pricing (handle variations)
        pricing = None
        for key, value in MODEL_PRICING.items():
            if key.lower() in model_name.lower():
                pricing = value
                break

        if not pricing:
            logger.warning(
                f"Unknown model pricing for {model_name}, using default GPT-3.5 pricing"
            )
            pricing = MODEL_PRICING["gpt-3.5-turbo"]

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


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
        span: OpenTelemetry span
        model_name: Name of the model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        total_tokens: Total tokens (if provided, used for validation)
        custom_pricing: Optional custom pricing
    """
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

