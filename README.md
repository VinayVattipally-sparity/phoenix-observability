# Phoenix Observability

A unified observability SDK for LLM projects using Arize Phoenix and OpenTelemetry.

## Overview

`phoenix-observability` provides a comprehensive observability solution for LLM applications, integrating Phoenix for LLM-specific monitoring and OpenTelemetry for distributed tracing. It offers instrumentation wrappers for LLMs, RAG systems, agents, and pipelines with built-in support for cost tracking, latency monitoring, hallucination detection, and more.

## Features

- **OpenTelemetry Integration**: Full OTLP support for distributed tracing
- **Phoenix Integration**: Native support for Arize Phoenix observability platform
- **LLM Instrumentation**: Automatic instrumentation for LLM calls with cost and latency tracking
- **RAG Support**: Instrumentation for retrieval-augmented generation systems
- **Agent Support**: Wrapper for agent-based LLM applications
- **Pipeline Tracking**: End-to-end pipeline observability
- **Cost Tracking**: Automatic cost calculation for LLM API calls
- **Hallucination Detection**: Built-in hallucination detection capabilities
- **PII Safety**: Automatic PII detection and safety analysis
- **System Metrics**: CPU, memory, and GPU monitoring
- **Error Handling**: Comprehensive error tracking and reporting

## Installation

### Basic Installation

```bash
pip install phoenix-observability
```

### With Phoenix Support

```bash
pip install phoenix-observability[phoenix]
```

## Quick Start

```python
from phoenix_observability import init_observability, instrument_llm
from phoenix_observability.config import ObservabilityConfig

# Initialize observability
init_observability(
    service_name="my-llm-service",
    phoenix_endpoint="http://localhost:6006",
    environment="dev"
)

# Instrument your LLM calls
@instrument_llm
def my_llm_function(prompt: str):
    # Your LLM code here
    return response
```

## Configuration

Configuration is managed through environment variables or the `ObservabilityConfig` class:

```python
from phoenix_observability.config import ObservabilityConfig

config = ObservabilityConfig()
```

### Environment Variables

- `PHOENIX_ENDPOINT`: Phoenix server endpoint (default: `http://localhost:6006`)
- `ENVIRONMENT`: Deployment environment (default: `dev`)
- `ENABLE_GPU_TRACKING`: Enable GPU monitoring (default: `false`)
- `ENABLE_PII_TRACKING`: Enable PII detection (default: `true`)
- `ENABLE_COST_TRACKING`: Enable cost tracking (default: `true`)
- `SERVICE_NAME`: Default service name (default: `phoenix_observability`)
- `OTLP_ENDPOINT`: Custom OTLP endpoint (optional)
- `BATCH_TIMEOUT_MS`: Batch export timeout (default: `5000`)
- `MAX_EXPORT_BATCH_SIZE`: Maximum batch size (default: `512`)

## Usage Examples

### LLM Instrumentation

```python
from phoenix_observability import instrument_llm

@instrument_llm
def call_openai(prompt: str):
    # Your OpenAI call here
    pass
```

### RAG Instrumentation

```python
from phoenix_observability import instrument_retriever

@instrument_retriever
def retrieve_documents(query: str):
    # Your retrieval logic here
    pass
```

### Agent Instrumentation

```python
from phoenix_observability import instrument_agent

@instrument_agent
def my_agent_function(input_data):
    # Your agent logic here
    pass
```

### Pipeline Instrumentation

```python
from phoenix_observability import instrument_pipeline

@instrument_pipeline
def my_pipeline(input_data):
    # Your pipeline logic here
    pass
```

## Package Structure

```
phoenix_observability/
├── __init__.py
├── config.py
├── otel_setup.py
├── phoenix_session.py
├── instrumentation/
│   ├── llm_wrapper.py
│   ├── rag_wrapper.py
│   ├── agent_wrapper.py
│   ├── pipeline_wrapper.py
│   ├── error_handler.py
│   └── structured_output.py
├── utils/
│   ├── cost_tracker.py
│   ├── latency.py
│   ├── hallucination.py
│   ├── accuracy.py
│   ├── pii_safety.py
│   ├── system_metrics.py
│   ├── gpu_monitor.py
│   ├── sanitize.py
│   └── pipeline_tracker.py
└── logging/
    ├── span_utils.py
    └── enrich.py
```

## Requirements

- Python 3.9+
- OpenTelemetry SDK 1.25+
- OpenTelemetry OTLP Exporter 1.25+
- python-dotenv 1.0.0+
- psutil 5.9.5+

Optional:
- arize-phoenix 2.5.0+ (for Phoenix UI support)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

