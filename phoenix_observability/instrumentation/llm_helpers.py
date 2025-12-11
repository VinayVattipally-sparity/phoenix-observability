"""
Helper functions for LLM instrumentation.

This module contains helper functions to break down the large instrument_llm()
wrapper into smaller, more maintainable pieces.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from opentelemetry import trace

logger = logging.getLogger(__name__)


def set_span_metadata(
    span: trace.Span,
    project_name: str,
    service_name: str,
    model_name: str,
    function_name: str,
) -> None:
    """
    Set basic metadata attributes on the span.

    Args:
        span: OpenTelemetry span to set attributes on.
        project_name: Project name.
        service_name: Service name.
        model_name: Model name.
        function_name: Function name being instrumented.
    """
    # Set OpenInference span kind
    span.set_attribute("openinference.span.kind", "LLM")

    # Set project and service attributes
    if project_name:
        span.set_attribute("project.name", project_name)
        span.set_attribute("project.id", project_name)
    if service_name:
        span.set_attribute("service.name", service_name)

    # Track API hit
    span.set_attribute("llm.api_hit", True)
    span.set_attribute("api.hit", True)

    # Basic LLM attributes
    span.set_attribute("llm.model_name", model_name)
    span.set_attribute("llm.model", model_name)  # Backward compatibility
    span.set_attribute("llm.function", function_name)

    # Extract and set vendor from model name
    vendor = _extract_vendor_from_model(model_name)
    span.set_attribute("llm.vendor", vendor)


def _extract_vendor_from_model(model_name: str) -> str:
    """
    Extract vendor name from model name.

    Args:
        model_name: Name of the model.

    Returns:
        Vendor name (openai, anthropic, google, meta, or openai as default).
    """
    model_lower = model_name.lower()
    if "gpt" in model_lower or "openai" in model_lower:
        return "openai"
    elif "claude" in model_lower or "anthropic" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower or "google" in model_lower:
        return "google"
    elif "llama" in model_lower:
        return "meta"
    return "openai"  # default


def extract_prompt_from_args(
    args: tuple,
    kwargs: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Extract prompt and user query from function arguments.

    Args:
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        Tuple of (prompt, user_query).
    """
    prompt = None
    user_query = None

    # Try positional argument first
    if args and len(args) > 0:
        prompt = args[0]
        user_query = args[0] if isinstance(args[0], str) else None
    # Try common keyword arguments
    elif "prompt" in kwargs:
        prompt = kwargs["prompt"]
        user_query = kwargs["prompt"] if isinstance(kwargs["prompt"], str) else None
    elif "messages" in kwargs:
        messages = kwargs["messages"]
        prompt = messages
        # Extract user query from messages
        if isinstance(messages, list) and messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict) and last_msg.get("role") == "user":
                user_query = last_msg.get("content", "")
    elif "query" in kwargs:
        user_query = kwargs["query"]
        prompt = kwargs["query"]

    return prompt, user_query


def extract_generation_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract generation configuration from kwargs.

    Args:
        kwargs: Keyword arguments.

    Returns:
        Dictionary with generation config attributes to set.
    """
    config_attrs = {}
    if "temperature" in kwargs:
        config_attrs["llm.generation_config.temperature"] = kwargs["temperature"]
    if "max_tokens" in kwargs or "max_output_tokens" in kwargs:
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_output_tokens")
        config_attrs["llm.generation_config.max_output_tokens"] = max_tokens
    if "operation" in kwargs:
        config_attrs["llm.operation"] = kwargs["operation"]
    return config_attrs


def extract_response_data(
    result: Any,
    model_name: str,
) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """
    Extract response text, usage data, and actual model name from LLM result.

    Args:
        result: LLM function result.
        model_name: Default model name.

    Returns:
        Tuple of (response_text, usage_data, actual_model_name).
    """
    actual_model_name = model_name
    response_text = None
    usage_data = None

    # Handle Google Gemini GenerateContentResponse object
    if hasattr(result, 'text') and hasattr(result, 'candidates'):
        try:
            response_text = result.text
            if hasattr(result, 'usage_metadata'):
                um = result.usage_metadata
                if um:
                    usage_data = {
                        "prompt_tokens": getattr(um, 'prompt_token_count', 0),
                        "completion_tokens": getattr(um, 'candidates_token_count', 0),
                        "total_tokens": getattr(um, 'total_token_count', None)
                    }
            if hasattr(result, 'model') and result.model:
                actual_model_name = result.model
            elif hasattr(result, 'model_name') and result.model_name:
                actual_model_name = result.model_name
        except Exception as e:
            logger.debug(f"Error extracting Gemini response: {e}")

    # Handle OpenAI ChatCompletion object
    elif hasattr(result, 'model'):
        actual_model_name = result.model
        if hasattr(result, 'choices') and result.choices:
            response_text = result.choices[0].message.content
            if hasattr(result, 'usage'):
                usage_data = {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens
                }

    # Handle dict format
    elif isinstance(result, dict):
        if "choices" in result and result["choices"]:
            response_text = result["choices"][0].get("message", {}).get("content", "")
        elif "content" in result:
            response_text = result["content"]
        elif "text" in result:
            response_text = result["text"]
        elif "output" in result:
            response_text = result["output"]

        if "model" in result:
            actual_model_name = result["model"]
        if "usage" in result:
            usage_data = result["usage"]

    # Handle string format
    elif isinstance(result, str):
        response_text = result

    return response_text or "", usage_data, actual_model_name


def extract_rag_context(
    func: Any,
    args: tuple,
    kwargs: Dict[str, Any],
) -> Optional[str]:
    """
    Extract RAG context from function arguments or signature.

    Args:
        func: Function being instrumented.
        args: Positional arguments.
        kwargs: Keyword arguments.

    Returns:
        RAG context as string, or None if not found.
    """
    # First try kwargs
    rag_context = kwargs.get("context") or kwargs.get("rag_context")
    if rag_context:
        return _convert_context_to_string(rag_context)

    # Try to get from positional args by inspecting signature
    try:
        import inspect
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        param_offset = 1 if param_names and param_names[0] == 'self' else 0

        # Look for 'context' or 'rag_context' in parameter names
        context_param_idx = None
        for i, param_name in enumerate(param_names[param_offset:], start=param_offset):
            if param_name in ['context', 'rag_context']:
                context_param_idx = i - param_offset
                break

        # Get from args if found
        if context_param_idx is not None and len(args) > context_param_idx:
            rag_context = args[context_param_idx]
            return _convert_context_to_string(rag_context)

        # Fallback: try second positional arg
        if len(args) > 1 and len(param_names) > 1 + param_offset:
            second_param = param_names[1 + param_offset]
            if any(keyword in second_param.lower() for keyword in ['context', 'doc', 'rag']):
                rag_context = args[1]
                return _convert_context_to_string(rag_context)
            elif isinstance(args[1], list):
                rag_context = args[1]
                return _convert_context_to_string(rag_context)
    except Exception as e:
        logger.debug(f"Could not inspect function signature: {e}")

    return None


def _convert_context_to_string(context: Any) -> str:
    """
    Convert context to string format.

    Args:
        context: Context to convert.

    Returns:
        Context as string.
    """
    if isinstance(context, list):
        return "\n".join(str(doc) for doc in context if doc)
    elif not isinstance(context, str):
        return str(context)
    return context

