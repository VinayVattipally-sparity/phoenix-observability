# Phoenix Observability

A unified observability SDK for LLM projects using Arize Phoenix and OpenTelemetry.

## Overview

`phoenix-observability` provides a comprehensive observability solution for LLM applications, integrating Phoenix for LLM-specific monitoring and OpenTelemetry for distributed tracing. It offers instrumentation wrappers for LLMs, RAG systems, agents, and pipelines with built-in support for cost tracking, latency monitoring, hallucination detection, and more.

## Features

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

## Installation

### Basic Installation

```bash
pip install phoenix-observability
```

### With Phoenix Support

```bash
pip install phoenix-observability[phoenix]
```

### With Development Dependencies

```bash
pip install phoenix-observability[test,lint,docs]
```

## Quick Start

**1. Create a `.env` file in your project root:**

```env
PHOENIX_ENDPOINT=https://phoenix-sparity.com
ENVIRONMENT=dev
SERVICE_NAME=my-llm-service
```

**2. Use the package in your code:**

```python
from phoenix_observability import init_observability, instrument_llm

# Initialize observability (reads from .env file)
init_observability(service_name="my-llm-service")

# Instrument your LLM calls
@instrument_llm
def my_llm_function(prompt: str):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

## Usage Examples

### LLM Instrumentation

```python
from phoenix_observability import instrument_llm

@instrument_llm(model_name="gpt-4", track_cost=True)
def call_openai(prompt: str):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### RAG Instrumentation

```python
from phoenix_observability import instrument_retriever

@instrument_retriever(log_documents=True)
def retrieve_documents(query: str):
    # Your retrieval logic here
    results = vector_db.search(query, top_k=5)
    return results
```

### Agent Instrumentation

```python
from phoenix_observability import instrument_agent

@instrument_agent(log_tool_inputs=True, log_tool_outputs=True)
def my_agent(input_data):
    # Your agent logic here
    tools = get_available_tools()
    result = agent.run(input_data, tools=tools)
    return result
```

### Pipeline Instrumentation

```python
from phoenix_observability import instrument_pipeline

@instrument_pipeline(pipeline_name="rag-pipeline")
def complete_rag_pipeline(query: str):
    docs = retrieve_documents(query)
    context = format_context(docs)
    response = generate_response(context, query)
    return response
```

For more examples, see the [Examples Documentation](docs/examples.md).

## Configuration

Configuration is managed through environment variables. See the [Configuration Guide](docs/getting-started/configuration.md) for complete details.

### Key Environment Variables

- `PHOENIX_ENDPOINT`: Phoenix server endpoint (**required**)
- `SERVICE_NAME`: Default service name (default: `phoenix_observability`)
- `ENVIRONMENT`: Deployment environment (default: `dev`)
- `ENABLE_COST_TRACKING`: Enable cost tracking (default: `true`)
- `ENABLE_PII_TRACKING`: Enable PII detection (default: `true`)
- `MAX_QUEUE_SIZE`: Maximum queue size (default: `2048`)
- `RATE_LIMIT_REQUESTS_PER_SECOND`: Rate limit per second (default: `10`)

See [Configuration Guide](docs/getting-started/configuration.md) for all options.

## Documentation

- **[Getting Started](docs/getting-started/quick-start.md)** - Quick setup guide
- **[Configuration Guide](docs/getting-started/configuration.md)** - Complete configuration reference
- **[Usage Examples](docs/examples.md)** - Comprehensive code examples
- **[Architecture](docs/architecture.md)** - System architecture and design
- **[API Reference](docs/api/core/config.md)** - Complete API documentation
- **[Security Guide](SECURITY.md)** - Security best practices
- **[CHANGELOG](CHANGELOG.md)** - Version history and changes

### Building Documentation

```bash
# Install docs dependencies
pip install phoenix-observability[docs]

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Package Structure

```
phoenix_observability/
├── __init__.py
├── config.py                 # Configuration management
├── otel_setup.py             # OpenTelemetry setup
├── phoenix_session.py        # Phoenix session management
├── instrumentation/          # Instrumentation decorators
│   ├── llm_wrapper.py
│   ├── rag_wrapper.py
│   ├── agent_wrapper.py
│   ├── pipeline_wrapper.py
│   ├── error_handler.py
│   └── structured_output.py
├── utils/                    # Utility modules
│   ├── cost_tracker.py
│   ├── rate_limiter.py
│   ├── http_client.py
│   ├── sanitize.py
│   ├── security.py
│   └── ...
└── logging/                  # Logging utilities
    └── structured.py
```

## Requirements

- Python 3.9+
- OpenTelemetry SDK 1.25+
- OpenTelemetry OTLP Exporter 1.25+
- python-dotenv 1.0.0+
- psutil 5.9.5+

Optional:
- arize-phoenix 2.5.0+ (for Phoenix UI support)
- openai 1.0.0+ (for OpenAI integrations)
- anthropic 0.7.0+ (for Anthropic integrations)
- google-generativeai 0.3.0+ (for Gemini integrations)

## Security

For security best practices, API key management, and secure deployment guidelines, see [SECURITY.md](SECURITY.md).

**Key Security Features:**
- ✅ API key format validation
- ✅ Input sanitization for URLs and names
- ✅ Automatic redaction of sensitive data from error messages
- ✅ Secure connections by default (`OTLP_INSECURE=false`)
- ✅ Comprehensive security documentation

## Testing

```bash
# Install test dependencies
pip install phoenix-observability[test]

# Run tests
pytest

# Run with coverage
pytest --cov=phoenix_observability --cov-report=html
```

## Performance Benchmarks

```bash
# Install benchmark dependencies
pip install phoenix-observability[benchmark]

# Run benchmarks
pytest benchmarks/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run linting and type checking
6. Submit a pull request

## License

MIT

## Support

For issues, questions, or contributions:
- GitHub Issues: [Create an issue](https://github.com/VinayVattipally-sparity/phoenix-observability/issues)
- Documentation: [Full Documentation](docs/index.md)
