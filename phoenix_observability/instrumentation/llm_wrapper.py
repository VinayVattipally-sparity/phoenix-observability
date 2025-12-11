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
from phoenix_observability.instrumentation.llm_helpers import (
    extract_generation_config,
    extract_prompt_from_args,
    extract_rag_context,
    extract_response_data,
    set_span_metadata,
)
from phoenix_observability.instrumentation.span_helpers import (
    extract_project_and_service_name,
)
from phoenix_observability.instrumentation.structured_output import (
    parse_and_validate_json,
)
from phoenix_observability.utils.metrics import (
    record_span_created,
    record_span_latency,
    record_error,
)

logger = logging.getLogger(__name__)


def instrument_llm(
    model_name: Optional[str] = None,
    track_cost: Optional[bool] = None,
    track_pii: Optional[bool] = None,
    expected_schema: Optional[Dict[str, Any]] = None,
    service_name: Optional[str] = None,
) -> Callable[[Callable], Callable]:
    """
    Decorator to instrument LLM calls with automatic observability.

    This decorator automatically creates OpenTelemetry spans for LLM function calls,
    tracking prompts, responses, latency, cost, and optional evaluations (PII detection,
    hallucination detection, accuracy evaluation).

    Args:
        model_name: Name of the LLM model (e.g., "gpt-4", "claude-3-5-sonnet").
            If not provided, the decorator will attempt to extract the model name
            from the function's return value or arguments. Supported formats include
            OpenAI, Anthropic, and Gemini response objects.
        track_cost: Whether to automatically calculate and track the cost of the LLM call.
            If None, uses the value from config.enable_cost_tracking. Cost is calculated
            based on input/output tokens and model pricing. Defaults to None (uses config).
        track_pii: Whether to run PII detection and safety analysis on the prompt and response.
            If None, uses the value from config.enable_pii_tracking. When enabled, detects
            personally identifiable information and calculates toxicity scores. Defaults to None (uses config).
        expected_schema: Optional dictionary defining the expected JSON schema for structured output.
            Keys should be field names and values should be Python types (e.g., {"name": str, "age": int}).
            The decorator will validate the response against this schema and log violations.
            Can also be a Pydantic BaseModel class for more complex validation.
        service_name: Optional service name for this specific LLM call. If provided, overrides
            the resource-level service.name attribute for this span. Useful for distinguishing
            different LLM services within the same application.

    Returns:
        A decorator function that wraps the original function with observability instrumentation.
        The wrapped function maintains the same signature and return value as the original.

    Example:
        Basic usage::

            @instrument_llm
            def chat(prompt: str):
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content

        With model name and cost tracking::

            @instrument_llm(model_name="gpt-4", track_cost=True)
            def expensive_call(prompt: str):
                # Cost will be automatically calculated
                return llm_call(prompt)

        With structured output validation::

            @instrument_llm(expected_schema={"name": str, "age": int})
            def extract_user_info(text: str):
                # Response will be validated against schema
                return llm_call(text)
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
                # Record span creation metric
                record_span_created("llm")
                
                # Extract project and service names
                project_name, final_service_name = extract_project_and_service_name(service_name)
                
                # Set span metadata
                set_span_metadata(span, project_name, final_service_name, func_model_name, func.__name__)

                timer = LatencyTimer()
                timer.start()

                try:
                    # Extract prompt and user query
                    prompt, user_query = extract_prompt_from_args(args, kwargs)

                    # Log prompt (sanitized)
                    if prompt:
                        sanitized_prompt = sanitize_prompt(prompt)
                        span.set_attribute("llm.prompt", sanitized_prompt)
                        span.set_attribute("input.value", sanitized_prompt[:1000])
                    
                    # Log user query separately
                    if user_query:
                        span.set_attribute("llm.user_query", user_query)
                    
                    # Extract and set generation config
                    gen_config = extract_generation_config(kwargs)
                    for key, value in gen_config.items():
                        span.set_attribute(key, value)

                    # Call the LLM function
                    result = func(*args, **kwargs)

                    # Measure latency
                    latency = timer.stop()
                    latency_ms = latency * 1000  # Convert to milliseconds
                    span.set_attribute("llm.latency_ms", latency_ms)
                    # Keep old name for backward compatibility
                    span.set_attribute("llm.latency_seconds", latency)
                    
                    # Record latency metric
                    record_span_latency(latency_ms, "llm")

                    # Extract response data
                    response_text, usage_data, actual_model_name = extract_response_data(result, func_model_name)
                    
                    # Update span with actual model name if different
                    if actual_model_name != func_model_name:
                        span.set_attribute("llm.model_name", actual_model_name)
                        span.set_attribute("llm.model", actual_model_name)
                        # Update vendor based on actual model
                        from phoenix_observability.instrumentation.llm_helpers import _extract_vendor_from_model
                        vendor = _extract_vendor_from_model(actual_model_name)
                        span.set_attribute("llm.vendor", vendor)

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
                    
                    # Check prompt injection on the INPUT prompt (not response)
                    prompt_injection_result = {}
                    prompt_injection_detected = False
                    if should_track_pii and sanitized_prompt and isinstance(sanitized_prompt, str):
                        from phoenix_observability.utils.pii_safety import detect_prompt_injection
                        # Enable warnings by default (can be controlled via config)
                        enable_warnings = getattr(config, 'enable_jailbreak_warnings', True)
                        prompt_injection_result = detect_prompt_injection(sanitized_prompt, enable_warnings=enable_warnings)
                        
                        # Set prompt injection attributes
                        prompt_injection_detected = prompt_injection_result.get("prompt_injection_detected", False)
                        span.set_attribute("llm.prompt_injection.detected", prompt_injection_detected)
                        span.set_attribute("prompt_injection.detected", prompt_injection_detected)
                        
                        # Always set prompt injection attributes (even if not detected)
                        span.set_attribute("llm.prompt_injection.confidence", prompt_injection_result.get("confidence", 0.0))
                        span.set_attribute("llm.prompt_injection.risk_level", prompt_injection_result.get("risk_level", "low"))
                        
                        # Set jailbreak-specific attributes
                        is_jailbreak = prompt_injection_result.get("is_jailbreak", False)
                        high_risk = prompt_injection_result.get("high_risk", False)
                        span.set_attribute("llm.prompt_injection.is_jailbreak", is_jailbreak)
                        span.set_attribute("llm.prompt_injection.high_risk", high_risk)
                        span.set_attribute("safety.prompt_injection.is_jailbreak", is_jailbreak)
                        span.set_attribute("safety.prompt_injection.high_risk", high_risk)
                        
                        patterns = prompt_injection_result.get("patterns_matched", [])
                        if patterns:
                            span.set_attribute("llm.prompt_injection.patterns", ",".join(patterns))
                            span.set_attribute("llm.prompt_injection.pattern_count", len(patterns))
                        else:
                            span.set_attribute("llm.prompt_injection.pattern_count", 0)
                        
                        # Add event for high-risk jailbreaks
                        if high_risk or is_jailbreak:
                            span.add_event(
                                "jailbreak_detected",
                                {
                                    "risk_level": prompt_injection_result.get("risk_level", "high"),
                                    "confidence": prompt_injection_result.get("confidence", 0.0),
                                    "patterns": ",".join(patterns) if patterns else "",
                                }
                            )
                    elif should_track_pii:
                        # Initialize with default values if prompt not available
                        prompt_injection_result = {
                            "prompt_injection_detected": False,
                            "confidence": 0.0,
                            "patterns_matched": [],
                            "risk_level": "low",
                        }
                        span.set_attribute("llm.prompt_injection.detected", False)
                        span.set_attribute("prompt_injection.detected", False)
                        span.set_attribute("llm.prompt_injection.confidence", 0.0)
                        span.set_attribute("llm.prompt_injection.risk_level", "low")
                        span.set_attribute("llm.prompt_injection.pattern_count", 0)
                    
                    if should_track_pii and response_text and isinstance(response_text, str):
                        # Use toxicity detection method from config
                        toxicity_method = config.toxicity_detection_method if hasattr(config, 'toxicity_detection_method') else None
                        safety_flags = analyze_safety(response_text, toxicity_method=toxicity_method, check_prompt_injection=False)
                        
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
                        
                        # Track SQL/XSS injection (different from prompt injection)
                        injection_detected = safety_flags.get("injection_detected", False)
                        
                        # Get prompt injection result (already checked above)
                        prompt_injection_detected = prompt_injection_result.get("prompt_injection_detected", False) if prompt_injection_result else False
                        
                        # Track safety concerns (include prompt injection)
                        has_concerns = (
                            pii_detected 
                            or injection_detected 
                            or prompt_injection_detected
                            or safety_flags.get("toxicity_score", 0) > 0.5
                        )
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
                        span.set_attribute("safety.prompt_injection_detected", prompt_injection_detected)
                        
                        # Set comprehensive prompt injection attributes in safety section (always set)
                        if prompt_injection_result:
                            span.set_attribute("safety.prompt_injection.detected", prompt_injection_detected)
                            span.set_attribute("safety.prompt_injection.confidence", prompt_injection_result.get("confidence", 0.0))
                            span.set_attribute("safety.prompt_injection.risk_level", prompt_injection_result.get("risk_level", "low"))
                            patterns = prompt_injection_result.get("patterns_matched", [])
                            if patterns:
                                span.set_attribute("safety.prompt_injection.patterns", ",".join(patterns))
                        else:
                            # Set default values if prompt injection wasn't checked
                            span.set_attribute("safety.prompt_injection.detected", False)
                            span.set_attribute("safety.prompt_injection.confidence", 0.0)
                            span.set_attribute("safety.prompt_injection.risk_level", "low")
                            # Set default values if prompt injection wasn't checked
                            span.set_attribute("safety.prompt_injection.detected", False)
                            span.set_attribute("safety.prompt_injection.confidence", 0.0)
                            span.set_attribute("safety.prompt_injection.risk_level", "low")

                    # Track structured output if schema provided
                    if expected_schema and response_text and isinstance(response_text, str):
                        parsed = parse_and_validate_json(span, response_text, expected_schema)
                        if parsed:
                            span.set_attribute("llm.structured_output.success", True)
                        else:
                            span.set_attribute("llm.structured_output.success", False)

                    # Hallucination detection (if RAG context is available)
                    try:
                        rag_context = extract_rag_context(func, args, kwargs)
                        
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
                    # Record error metric
                    record_error(type(e).__name__)
                    handle_error(span, e, {"function": func.__name__})
                    raise

        return wrapper

    return decorator

