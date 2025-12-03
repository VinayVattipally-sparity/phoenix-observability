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
        model_lower = model_name.lower()
        
        # First, try exact match
        if model_lower in MODEL_PRICING:
            pricing = MODEL_PRICING[model_lower]
        else:
            # Try partial matches (handle version suffixes, dates, etc.)
            # Sort by key length (longer = more specific) to prefer exact matches
            sorted_keys = sorted(MODEL_PRICING.keys(), key=len, reverse=True)
            
            for key in sorted_keys:
                key_lower = key.lower()
                # Check if key is contained in model name or vice versa
                # Also handle common variations like "gemini-2.0-flash-exp" matching "gemini-2.0-flash"
                if key_lower in model_lower or model_lower in key_lower:
                    # Additional check: ensure we're matching the right model family
                    # e.g., "gemini-2.0" should match "gemini-2.0-flash" but not "gemini-1.5"
                    key_parts = key_lower.split("-")
                    model_parts = model_lower.split("-")
                    
                    # Match if major parts align (e.g., "gemini", "2.0", "flash")
                    if len(key_parts) >= 2:
                        if key_parts[0] in model_parts and key_parts[1] in model_parts:
                            pricing = MODEL_PRICING[key]
                            break
                    else:
                        # For simple names, just check containment
                        pricing = MODEL_PRICING[key]
                        break

        if not pricing:
            # Try to infer provider and use provider-specific default
            provider_default = None
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
