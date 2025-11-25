"""
Phoenix Observability Package

A shared observability SDK for LLM projects using Arize Phoenix and OpenTelemetry.
"""

from phoenix_observability.config import ObservabilityConfig
from phoenix_observability.otel_setup import init_observability
from phoenix_observability.instrumentation.llm_wrapper import instrument_llm
from phoenix_observability.instrumentation.rag_wrapper import instrument_retriever
from phoenix_observability.instrumentation.agent_wrapper import instrument_agent
from phoenix_observability.instrumentation.pipeline_wrapper import instrument_pipeline

__version__ = "0.1.0"

__all__ = [
    "init_observability",
    "instrument_llm",
    "instrument_retriever",
    "instrument_agent",
    "instrument_pipeline",
    "ObservabilityConfig",
]

