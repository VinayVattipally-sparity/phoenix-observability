# Phoenix Observability

A unified observability SDK for LLM projects using Arize Phoenix and OpenTelemetry.

## Overview

`phoenix-observability` provides a comprehensive observability solution for LLM applications, integrating Phoenix for LLM-specific monitoring and OpenTelemetry for distributed tracing. It offers instrumentation wrappers for LLMs, RAG systems, agents, and pipelines with built-in support for cost tracking, latency monitoring, hallucination detection, and more.

## Key Features

- **🔍 OpenTelemetry Integration**: Full OTLP support for distributed tracing
- **🦅 Phoenix Integration**: Native support for Arize Phoenix observability platform
- **🤖 LLM Instrumentation**: Automatic instrumentation for LLM calls with cost and latency tracking
- **📚 RAG Support**: Instrumentation for retrieval-augmented generation systems
- **🤝 Agent Support**: Wrapper for agent-based LLM applications
- **🔄 Pipeline Tracking**: End-to-end pipeline observability
- **💰 Cost Tracking**: Automatic cost calculation for LLM API calls
- **🎭 Hallucination Detection**: Built-in hallucination detection capabilities
- **🔒 PII Safety**: Automatic PII detection and safety analysis
- **📊 System Metrics**: CPU, memory, and GPU monitoring
- **⚠️ Error Handling**: Comprehensive error tracking and reporting
- **⚡ Rate Limiting**: Built-in rate limiting for external API calls
- **🌐 Connection Pooling**: HTTP connection pooling for better performance
- **⚙️ Configurable**: Highly configurable via environment variables

## Quick Start

```python
from phoenix_observability import init_observability, instrument_llm

# Initialize observability
init_observability(service_name="my-llm-service")

# Instrument your LLM calls
@instrument_llm
def my_llm_function(prompt: str):
    # Your LLM code here
    return response
```

## Installation

```bash
pip install phoenix-observability
```

For Phoenix UI support:

```bash
pip install phoenix-observability[phoenix]
```

## Documentation

- [Getting Started](getting-started/quick-start.md) - Quick setup guide
- [User Guide](user-guide/llm-instrumentation.md) - Detailed usage instructions
- [API Reference](api/core/config.md) - Complete API documentation
- [Architecture](architecture.md) - System architecture and design
- [Examples](examples.md) - Code examples and use cases
- [Security](security.md) - Security best practices

## Requirements

- Python 3.9+
- OpenTelemetry SDK 1.25+
- OpenTelemetry OTLP Exporter 1.25+
- python-dotenv 1.0.0+
- psutil 5.9.5+

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

