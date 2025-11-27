"""
LLM instrumentation wrapper.

Decorator for automatically logging LLM calls with prompt, response, latency, cost, etc.
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import SpanKind

from phoenix_observability.config import get_config
from phoenix_observability.otel_setup import get_tracer
from phoenix_observability.utils.cost_tracker import attach_cost_to_span
from phoenix_observability.utils.latency import LatencyTimer
from phoenix_observability.utils.pii_safety import analyze_safety
from phoenix_observability.utils.sanitize import sanitize_prompt, sanitize_response
from phoenix_observability.utils.hallucination import judge_hallucination
from phoenix_observability.utils.accuracy import evaluate_accuracy
from phoenix_observability.utils.system_metrics import attach_system_metrics_to_span
from phoenix_observability.instrumentation.error_handler import handle_error
from phoenix_observability.instrumentation.structured_output import (
    parse_and_validate_json,
)

logger = logging.getLogger(__name__)


def instrument_llm(
    model_name: Optional[str] = None,
    track_cost: Optional[bool] = None,
    track_pii: Optional[bool] = None,
    expected_schema: Optional[Dict[str, Any]] = None,
):
    """
    Decorator to instrument LLM calls.

    Args:
        model_name: Name of the model (if not provided, will try to extract from function)
        track_cost: Whether to track cost (defaults to config)
        track_pii: Whether to track PII and safety flags (defaults to config)
        expected_schema: Optional expected JSON schema for structured output

    Returns:
        Decorated function
    """
    config = get_config()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            func_model_name = model_name or func.__name__

            # Create span
            with tracer.start_as_current_span(
                f"llm.call.{func.__name__}"
            ) as span:
                # Set OpenInference span kind
                span.set_attribute("openinference.span.kind", "LLM")
                
                # Track API hit (indicates an API call was made)
                span.set_attribute("llm.api_hit", True)
                span.set_attribute("api.hit", True)  # Also set flat for dashboard
                
                # Basic LLM attributes
                span.set_attribute("llm.model_name", func_model_name)
                span.set_attribute("llm.model", func_model_name)  # Keep for backward compatibility
                span.set_attribute("llm.function", func.__name__)
                
                # Extract vendor from model name
                vendor = "openai"  # default
                model_lower = func_model_name.lower()
                if "gpt" in model_lower or "openai" in model_lower:
                    vendor = "openai"
                elif "claude" in model_lower or "anthropic" in model_lower:
                    vendor = "anthropic"
                elif "gemini" in model_lower or "google" in model_lower:
                    vendor = "google"
                elif "llama" in model_lower:
                    vendor = "meta"
                span.set_attribute("llm.vendor", vendor)

                timer = LatencyTimer()
                timer.start()

                try:
                    # Extract prompt from args/kwargs (common patterns)
                    prompt = None
                    user_query = None
                    if args and len(args) > 0:
                        prompt = args[0]
                        user_query = args[0] if isinstance(args[0], str) else None
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

                    # Log prompt (sanitized)
                    if prompt:
                        sanitized_prompt = sanitize_prompt(prompt)
                        span.set_attribute("llm.prompt", sanitized_prompt)
                        # Set input.value for OpenInference
                        span.set_attribute("input.value", sanitized_prompt[:1000])  # Truncate for display
                    
                    # Log user query separately
                    if user_query:
                        span.set_attribute("llm.user_query", user_query)
                    
                    # Extract generation config if available
                    if "temperature" in kwargs:
                        span.set_attribute("llm.generation_config.temperature", kwargs["temperature"])
                    if "max_tokens" in kwargs or "max_output_tokens" in kwargs:
                        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_output_tokens")
                        span.set_attribute("llm.generation_config.max_output_tokens", max_tokens)
                    
                    # Extract operation name if available
                    if "operation" in kwargs:
                        span.set_attribute("llm.operation", kwargs["operation"])

                    # Call the LLM function
                    result = func(*args, **kwargs)

                    # Measure latency
                    latency = timer.stop()
                    latency_ms = latency * 1000  # Convert to milliseconds
                    span.set_attribute("llm.latency_ms", latency_ms)
                    # Keep old name for backward compatibility
                    span.set_attribute("llm.latency_seconds", latency)

                    # Extract actual model name from response
                    actual_model_name = func_model_name
                    response_text = None
                    usage_data = None
                    
                    # Handle OpenAI ChatCompletion object
                    if hasattr(result, 'model'):
                        actual_model_name = result.model
                        span.set_attribute("llm.model_name", actual_model_name)
                        span.set_attribute("llm.model", actual_model_name)
                        # Update vendor based on actual model
                        model_lower = actual_model_name.lower()
                        if "gpt" in model_lower or "openai" in model_lower:
                            vendor = "openai"
                        elif "claude" in model_lower or "anthropic" in model_lower:
                            vendor = "anthropic"
                        elif "gemini" in model_lower or "google" in model_lower:
                            vendor = "google"
                        elif "llama" in model_lower:
                            vendor = "meta"
                        span.set_attribute("llm.vendor", vendor)
                    
                    # Extract response text
                    if hasattr(result, 'choices') and result.choices:
                        # OpenAI ChatCompletion format
                        response_text = result.choices[0].message.content
                        if hasattr(result, 'usage'):
                            usage_data = {
                                "prompt_tokens": result.usage.prompt_tokens,
                                "completion_tokens": result.usage.completion_tokens,
                                "total_tokens": result.usage.total_tokens
                            }
                    elif isinstance(result, dict):
                        # Dict format
                        if "choices" in result:
                            if result["choices"]:
                                response_text = result["choices"][0].get("message", {}).get("content", "")
                        elif "content" in result:
                            response_text = result["content"]
                        elif "text" in result:
                            response_text = result["text"]
                        elif "output" in result:
                            response_text = result["output"]
                        
                        # Extract model from dict if available
                        if "model" in result:
                            actual_model_name = result["model"]
                            span.set_attribute("llm.model_name", actual_model_name)
                            span.set_attribute("llm.model", actual_model_name)
                        
                        # Extract usage from dict
                        if "usage" in result:
                            usage_data = result["usage"]
                    elif isinstance(result, str):
                        response_text = result

                    # Log response (sanitized)
                    if response_text:
                        sanitized_response = sanitize_response(response_text)
                        span.set_attribute("llm.response", sanitized_response)
                        # Set output.value for OpenInference (truncate for display)
                        span.set_attribute("output.value", sanitized_response[:1000])
                    else:
                        # If we can't extract text, at least log the result object as string
                        result_str = str(result)[:1000]
                        span.set_attribute("llm.response", result_str)
                        span.set_attribute("output.value", result_str)

                    # Track cost and tokens if enabled
                    should_track_cost = (
                        track_cost if track_cost is not None else config.enable_cost_tracking
                    )
                    if should_track_cost and usage_data:
                        # Extract token usage
                        if isinstance(usage_data, dict):
                            input_tokens = usage_data.get("prompt_tokens", 0)
                            output_tokens = usage_data.get("completion_tokens", 0)
                            total_tokens = usage_data.get("total_tokens")
                        else:
                            # Handle object with attributes
                            input_tokens = getattr(usage_data, "prompt_tokens", 0)
                            output_tokens = getattr(usage_data, "completion_tokens", 0)
                            total_tokens = getattr(usage_data, "total_tokens", None)
                        
                        attach_cost_to_span(
                            span, actual_model_name, input_tokens, output_tokens, total_tokens
                        )

                    # Track PII and safety if enabled
                    should_track_pii = (
                        track_pii if track_pii is not None else config.enable_pii_tracking
                    )
                    if should_track_pii and response_text and isinstance(response_text, str):
                        # Use toxicity detection method from config
                        toxicity_method = config.toxicity_detection_method if hasattr(config, 'toxicity_detection_method') else None
                        safety_flags = analyze_safety(response_text, toxicity_method=toxicity_method)
                        
                        # Track PII with nested structure (matching previous implementation)
                        pii_detected = safety_flags.get("pii_detected", False)
                        span.set_attribute("llm.pii.detected", pii_detected)
                        
                        # Count PII types detected
                        pii_types = []
                        if safety_flags.get("email_detected"):
                            pii_types.append("EMAIL")
                        if safety_flags.get("phone_detected"):
                            pii_types.append("PHONE")
                        if safety_flags.get("ssn_detected"):
                            pii_types.append("SSN")
                        if safety_flags.get("credit_card_detected"):
                            pii_types.append("CREDIT_CARD")
                        
                        if pii_types:
                            span.set_attribute("llm.pii.types", ",".join(pii_types))
                            span.set_attribute("llm.pii.count", len(pii_types))
                        else:
                            span.set_attribute("llm.pii.count", 0)
                        
                        # Track prompt injection
                        injection_detected = safety_flags.get("injection_detected", False)
                        span.set_attribute("llm.prompt_injection.detected", injection_detected)
                        
                        # Track safety concerns
                        has_concerns = pii_detected or injection_detected or safety_flags.get("toxicity_score", 0) > 0.5
                        span.set_attribute("llm.safety.has_concerns", has_concerns)
                        span.set_attribute("llm.safety.blocked", False)  # Can be set based on moderation API response
                        
                        # Track toxicity score and details
                        toxicity_score = safety_flags.get("toxicity_score", 0.0)
                        toxicity_method = safety_flags.get("toxicity_method", "heuristic")
                        toxicity_flagged = safety_flags.get("toxicity_flagged", False)
                        toxicity_categories = safety_flags.get("toxicity_categories", {})
                        
                        span.set_attribute("llm.safety.toxicity_score", toxicity_score)
                        span.set_attribute("safety.toxicity_score", toxicity_score)  # Also set flat
                        span.set_attribute("safety.toxicity_method", toxicity_method)
                        span.set_attribute("safety.toxicity_flagged", toxicity_flagged)
                        
                        # Track detailed toxicity categories if available
                        if toxicity_categories:
                            for category, score in toxicity_categories.items():
                                span.set_attribute(f"safety.toxicity.{category}", float(score))
                        
                        # Keep old names for backward compatibility
                        span.set_attribute("safety.pii_detected", pii_detected)
                        span.set_attribute("safety.prompt_injection_detected", injection_detected)

                    # Track structured output if schema provided
                    if expected_schema and response_text and isinstance(response_text, str):
                        parsed = parse_and_validate_json(span, response_text, expected_schema)
                        if parsed:
                            span.set_attribute("llm.structured_output.success", True)
                        else:
                            span.set_attribute("llm.structured_output.success", False)

                    # Hallucination detection (if RAG context is available)
                    # Check if there's a parent span with rag.context
                    try:
                        from opentelemetry import trace as otel_trace
                        import inspect
                        current_span = otel_trace.get_current_span()
                        rag_context = None
                        
                        # First try to get context from kwargs (common pattern in RAG)
                        rag_context = kwargs.get("context") or kwargs.get("rag_context")
                        
                        # If not in kwargs, try to get from positional args
                        # Check function signature to find context parameter
                        if not rag_context:
                            try:
                                sig = inspect.signature(func)
                                param_names = list(sig.parameters.keys())
                                # Skip 'self' if it's a method
                                param_offset = 1 if param_names and param_names[0] == 'self' else 0
                                
                                # Look for 'context' or 'rag_context' in parameter names
                                context_param_idx = None
                                for i, param_name in enumerate(param_names[param_offset:], start=param_offset):
                                    if param_name in ['context', 'rag_context']:
                                        # Calculate the index in the args array (accounting for param_offset)
                                        context_param_idx = i - param_offset
                                        break
                                
                                # If found, get from args
                                if context_param_idx is not None and len(args) > context_param_idx:
                                    rag_context = args[context_param_idx]
                                # Fallback: try second positional arg if function has at least 2 params
                                elif len(args) > 1 and len(param_names) > 1 + param_offset:
                                    # Check if second param name suggests it's context
                                    second_param = param_names[1 + param_offset]
                                    if 'context' in second_param.lower() or 'doc' in second_param.lower() or 'rag' in second_param.lower():
                                        rag_context = args[1]
                                    # Or if it's a list (common for RAG contexts)
                                    elif isinstance(args[1], list):
                                        rag_context = args[1]
                            except Exception as e:
                                logger.debug(f"Could not inspect function signature: {e}")
                        
                        # If not in kwargs or args, try to get from parent span's rag.context attribute
                        # OpenTelemetry doesn't directly expose parent span attributes,
                        # but we can check the current span context for stored values
                        if not rag_context and current_span:
                            # Try to get rag.context from the span's context
                            # This would be set by the RAG retriever wrapper
                            # We'll check if it's available in the trace context
                            try:
                                # Access span context to get parent attributes
                                # Note: This is a workaround - ideally parent span attributes
                                # would be accessible, but OpenTelemetry doesn't expose them directly
                                pass
                            except:
                                pass
                        
                        # Convert context to string if it's a list
                        if rag_context:
                            if isinstance(rag_context, list):
                                rag_context = "\n".join(str(doc) for doc in rag_context if doc)
                            elif not isinstance(rag_context, str):
                                rag_context = str(rag_context)
                        
                        # Always create hallucination evaluation span (matching D:/Phoenix/anomaly_detection pattern)
                        # If context is available, run full evaluation; otherwise mark as "no context"
                        if response_text and isinstance(response_text, str):
                            # Create separate child span for hallucination evaluation
                            eval_tracer = get_tracer()
                            with eval_tracer.start_as_current_span(
                                "evaluation.hallucination", 
                                kind=SpanKind.INTERNAL
                            ) as eval_span:
                                # Set OpenInference span kind for Phoenix to recognize as EVALUATOR
                                eval_span.set_attribute("openinference.span.kind", "EVALUATOR")
                                eval_span.set_attribute("evaluation.type", "hallucination")
                                
                                # Set input and output values for Phoenix dashboard
                                # Include user query in input for better context
                                if user_query:
                                    eval_input = f"Query: {user_query[:500]}\n\nResponse: {response_text[:500]}"
                                else:
                                    eval_input = response_text[:1000]
                                eval_span.set_attribute("input.value", eval_input)
                                eval_span.set_attribute("output.value", response_text[:1000])
                                
                                # Always run hallucination detection (with or without context)
                                # If context available: compares against context
                                # If no context: uses LLM judge to evaluate internal consistency and plausibility
                                # Pass user_query for better evaluation when no context is available
                                hall_result = judge_hallucination(
                                    context=rag_context,  # Can be None
                                    answer=response_text,
                                    user_query=user_query,  # Pass user query for better evaluation
                                    use_llm=True  # Use LLM judge for evaluation
                                )
                                
                                # Set evaluation attributes on the evaluation span
                                eval_span.set_attribute("evaluation.hallucination.score", float(hall_result["score"]))
                                eval_span.set_attribute("evaluation.hallucination.is_hallucinating", bool(hall_result["hallucinates"]))
                                eval_span.set_attribute("evaluation.hallucination.flag", bool(hall_result["hallucinates"]))
                                
                                # Set explanation/reason
                                reason = hall_result.get("reason", "")
                                if reason:
                                    eval_span.set_attribute("evaluation.explanation", reason[:500])
                                    eval_span.set_attribute("evaluation.hallucination.explanation", reason[:500])
                                    eval_span.set_attribute("evaluation.hallucination.reason", reason[:500])
                                
                                # Also set on parent LLM span for backward compatibility
                                span.set_attribute("hallucination.score", hall_result["score"])
                                span.set_attribute("hallucination.flag", hall_result["hallucinates"])
                                span.set_attribute("hallucination.reason", reason)
                                span.set_attribute("evaluator.hallucination.score", hall_result["score"])
                                span.set_attribute("evaluator.hallucination.flag", hall_result["hallucinates"])
                                span.set_attribute("evaluator.hallucination.explanation", reason)
                                
                                eval_span.set_status(trace.Status(trace.StatusCode.OK))
                    except Exception as e:
                        logger.warning(f"Hallucination detection failed: {e}", exc_info=True)
                    
                    # Accuracy evaluation (if ground truth is provided)
                    # Create a separate child span for the evaluation
                    try:
                        ground_truth = kwargs.get("ground_truth") or kwargs.get("expected_answer")
                        
                        if ground_truth and response_text and isinstance(response_text, str):
                            # Convert ground truth to string if needed
                            if isinstance(ground_truth, list):
                                ground_truth = "\n".join(str(item) for item in ground_truth if item)
                            elif not isinstance(ground_truth, str):
                                ground_truth = str(ground_truth)
                            
                            # Create separate child span for accuracy evaluation
                            eval_tracer = get_tracer()
                            with eval_tracer.start_as_current_span(
                                "evaluation.accuracy", 
                                kind=SpanKind.INTERNAL
                            ) as eval_span:
                                # Set OpenInference span kind for Phoenix to recognize as EVALUATOR
                                eval_span.set_attribute("openinference.span.kind", "EVALUATOR")
                                eval_span.set_attribute("evaluation.type", "accuracy")
                                
                                # Set input and output values for Phoenix dashboard
                                eval_span.set_attribute("input.value", f"Ground Truth: {ground_truth[:500]}\nPredicted: {response_text[:500]}")
                                eval_span.set_attribute("output.value", response_text[:1000])
                                
                                # Run accuracy evaluation
                                accuracy_result = evaluate_accuracy(
                                    ground_truth=ground_truth,
                                    predicted=response_text
                                )
                                
                                # Set evaluation attributes on the evaluation span
                                eval_span.set_attribute("evaluation.accuracy.score", float(accuracy_result["score"]))
                                eval_span.set_attribute("evaluation.accuracy.is_correct", bool(accuracy_result["is_correct"]))
                                
                                # Set explanation/reason
                                reason = accuracy_result.get("reason", "")
                                if reason:
                                    eval_span.set_attribute("evaluation.explanation", reason[:500])
                                    eval_span.set_attribute("evaluation.accuracy.explanation", reason[:500])
                                    eval_span.set_attribute("evaluation.accuracy.reason", reason[:500])
                                
                                # Also set on parent LLM span for backward compatibility
                                span.set_attribute("accuracy.score", accuracy_result["score"])
                                span.set_attribute("accuracy.is_correct", accuracy_result["is_correct"])
                                span.set_attribute("accuracy.reason", reason)
                                span.set_attribute("evaluator.accuracy.score", accuracy_result["score"])
                                span.set_attribute("evaluator.accuracy.is_correct", accuracy_result["is_correct"])
                                span.set_attribute("evaluator.accuracy.explanation", reason)
                                
                                eval_span.set_status(trace.Status(trace.StatusCode.OK))
                    except Exception as e:
                        logger.debug(f"Accuracy evaluation skipped: {e}")
                    
                    # Attach system metrics (optional, can be disabled for performance)
                    try:
                        attach_system_metrics_to_span(span, include_gpu=config.enable_gpu_tracking)
                    except Exception as e:
                        logger.debug(f"System metrics tracking skipped: {e}")

                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result

                except Exception as e:
                    timer.stop()
                    handle_error(span, e, {"function": func.__name__})
                    raise

        return wrapper

    return decorator

