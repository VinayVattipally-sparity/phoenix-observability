"""
Agent instrumentation wrapper.

Wraps agent steps and tool calls to log tool invocations, inputs/outputs, and errors.
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace

from phoenix_observability.otel_setup import get_tracer
from phoenix_observability.utils.latency import LatencyTimer
from phoenix_observability.instrumentation.error_handler import handle_error

logger = logging.getLogger(__name__)


def instrument_agent(
    agent_name: Optional[str] = None,
    log_tool_inputs: bool = True,
    log_tool_outputs: bool = True,
    log_intermediate_steps: bool = False,
    service_name: Optional[str] = None,
):
    """
    Decorator to instrument agent operations.

    Args:
        agent_name: Name of the agent (defaults to function name)
        log_tool_inputs: Whether to log tool inputs
        log_tool_outputs: Whether to log tool outputs
        log_intermediate_steps: Whether to log intermediate reasoning steps
        service_name: Optional service name for this agent (overrides resource-level service.name)

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = agent_name or func.__name__

            # Create span
            with tracer.start_as_current_span(f"agent.step.{name}") as span:
                # Set OpenInference span kind for agents
                span.set_attribute("openinference.span.kind", "CHAIN")
                
                # Add project_name and service_name from resource or environment
                from phoenix_observability.config import get_config
                config = get_config()
                import os
                project_name = os.getenv("PHOENIX_PROJECT_NAME") or config.default_service_name
                # Use service_name from decorator parameter if provided, otherwise get from resource
                final_service_name = service_name  # Use decorator parameter if provided
                
                # Try to get from resource attributes if available (only if not provided in decorator)
                try:
                    from opentelemetry import trace as otel_trace
                    tracer_provider = otel_trace.get_tracer_provider()
                    if hasattr(tracer_provider, 'resource') and tracer_provider.resource:
                        resource_attrs = dict(tracer_provider.resource.attributes)
                        project_name = resource_attrs.get("project.name") or resource_attrs.get("project.id") or project_name
                        # Only use resource service.name if not provided in decorator parameter
                        if not final_service_name:
                            final_service_name = resource_attrs.get("service.name")
                except:
                    pass
                
                # Fallback to config default if still not set
                if not final_service_name:
                    final_service_name = config.default_service_name
                
                # Set project and service attributes on span
                if project_name:
                    span.set_attribute("project.name", project_name)
                    span.set_attribute("project.id", project_name)
                if final_service_name:
                    span.set_attribute("service.name", final_service_name)
                
                span.set_attribute("agent.step.name", name)
                # Keep old name for backward compatibility
                span.set_attribute("agent.name", name)
                span.set_attribute("agent.function", func.__name__)

                # Extract input if available
                input_data = None
                if args and len(args) > 0:
                    input_data = args[0]
                elif "input" in kwargs:
                    input_data = kwargs["input"]
                elif "query" in kwargs:
                    input_data = kwargs["query"]

                if input_data:
                    input_str = str(input_data)[:1000]  # Truncate
                    span.set_attribute("agent.step.input", input_str)
                    # Keep old name for backward compatibility
                    span.set_attribute("agent.input", input_str)

                timer = LatencyTimer()
                timer.start()

                try:
                    # Call the agent function
                    result = func(*args, **kwargs)

                    # Measure latency
                    latency = timer.stop()
                    latency_ms = latency * 1000  # Convert to milliseconds
                    span.set_attribute("agent.step.latency_ms", latency_ms)
                    # Keep old name for backward compatibility
                    span.set_attribute("agent.latency_seconds", latency)

                    # Extract and log result
                    if isinstance(result, dict):
                        # Common agent response formats
                        if "output" in result:
                            output = str(result["output"])[:2000]
                            span.set_attribute("agent.step.output", output)
                            # Keep old name for backward compatibility
                            span.set_attribute("agent.output", output)

                        # Log tool calls if present
                        if "tool_calls" in result or "intermediate_steps" in result:
                            tool_calls = result.get("tool_calls", [])
                            intermediate_steps = result.get("intermediate_steps", [])

                            if tool_calls:
                                span.set_attribute("agent.tool_calls_count", len(tool_calls))
                                for i, tool_call in enumerate(tool_calls[:10]):  # Limit to 10
                                    tool_name = tool_call.get("name", tool_call.get("tool", "unknown"))
                                    span.set_attribute(f"agent.tool.{i}.name", tool_name)

                                    if log_tool_inputs:
                                        tool_input = tool_call.get("input", tool_call.get("args", ""))
                                        input_str = str(tool_input)[:500]
                                        span.set_attribute(f"agent.tool.{i}.input", input_str)

                                    if log_tool_outputs:
                                        tool_output = tool_call.get("output", tool_call.get("result", ""))
                                        output_str = str(tool_output)[:1000]
                                        span.set_attribute(f"agent.tool.{i}.output", output_str)

                            if intermediate_steps and log_intermediate_steps:
                                span.set_attribute("agent.intermediate_steps_count", len(intermediate_steps))
                                for i, step in enumerate(intermediate_steps[:5]):  # Limit to 5
                                    step_str = str(step)[:500]
                                    span.set_attribute(f"agent.intermediate_step.{i}", step_str)

                        # Log errors if present
                        if "error" in result:
                            error = result["error"]
                            span.set_attribute("agent.error", str(error))
                            span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))

                    elif isinstance(result, str):
                        # Simple string output
                        output = result[:2000]
                        span.set_attribute("agent.step.output", output)
                        # Keep old name for backward compatibility
                        span.set_attribute("agent.output", output)

                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result

                except Exception as e:
                    timer.stop()
                    handle_error(span, e, {"agent": name, "function": func.__name__})
                    raise

        return wrapper

    return decorator

