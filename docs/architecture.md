# Architecture Documentation

This document describes the architecture and design of the Phoenix Observability package.

## Overview

Phoenix Observability is designed as a lightweight, configurable observability SDK that integrates Phoenix and OpenTelemetry for comprehensive LLM application monitoring. The architecture emphasizes modularity, extensibility, and performance.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (User's LLM Application Code)                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Uses decorators
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Instrumentation Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  LLM Wrapper │  │  RAG Wrapper │  │Agent Wrapper │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Shared Helpers & Utilities                  │   │
│  │  - Span Helpers  - Error Handler  - Helpers         │   │
│  └─────────────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Creates spans & attributes
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              OpenTelemetry Layer                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Tracer Provider & Span Processors            │  │
│  └──────────────────────┬───────────────────────────────┘  │
│                         │                                   │
│  ┌──────────────────────▼───────────────────────────────┐ │
│  │         OTLP Exporter (HTTP/gRPC)                      │ │
│  └──────────────────────┬───────────────────────────────┘ │
└─────────────────────────┼───────────────────────────────────┘
                          │
                          │ Exports traces
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Phoenix / OTLP Endpoint                        │
│         (Observability Backend)                             │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Configuration System

**Location**: `phoenix_observability/config.py`

The configuration system provides centralized, thread-safe configuration management:

- **ObservabilityConfig**: Main configuration class that loads settings from environment variables
- **Thread-safe singleton**: Uses double-checked locking pattern
- **Environment variable support**: All settings configurable via `.env` file
- **Type-safe**: Full type hints for all configuration options

**Key Features**:
- Default values for all settings
- Validation of configuration values
- Support for feature flags
- Rate limiting configuration
- HTTP client pooling configuration

### 2. OpenTelemetry Setup

**Location**: `phoenix_observability/otel_setup.py`

Handles OpenTelemetry initialization and configuration:

- **init_observability()**: Main initialization function
- **get_tracer()**: Get tracer instance for creating spans
- **Resource attributes**: Automatically sets service name, environment, etc.
- **Exporter configuration**: Supports both HTTP and gRPC OTLP exporters
- **Batch processing**: Configurable batch size and timeout

**Key Features**:
- Automatic Phoenix project creation (if Phoenix client available)
- Endpoint sanitization and validation
- Secure connections by default
- Configurable queue sizes and batch settings

### 3. Instrumentation Wrappers

#### LLM Wrapper

**Location**: `phoenix_observability/instrumentation/llm_wrapper.py`

Comprehensive instrumentation for LLM calls:

- **Automatic span creation**: Creates spans for each LLM call
- **Cost tracking**: Calculates and attaches cost information
- **Latency tracking**: Measures execution time
- **PII detection**: Optional PII and safety analysis
- **Hallucination detection**: Optional hallucination evaluation
- **Structured output validation**: Validates JSON responses against schemas

**Key Features**:
- Model name extraction from various LLM SDKs
- Prompt and response sanitization
- Usage data extraction (tokens, etc.)
- RAG context detection
- Error handling and reporting

#### RAG Wrapper

**Location**: `phoenix_observability/instrumentation/rag_wrapper.py`

Instrumentation for retrieval operations:

- **Document logging**: Logs retrieved documents
- **Metadata tracking**: Captures retrieval metadata
- **Latency measurement**: Tracks retrieval time
- **Context truncation**: Handles long contexts intelligently

#### Agent Wrapper

**Location**: `phoenix_observability/instrumentation/agent_wrapper.py`

Instrumentation for agent-based systems:

- **Tool call tracking**: Logs tool invocations
- **Input/output logging**: Captures tool inputs and outputs
- **Intermediate steps**: Optional logging of agent reasoning steps
- **Error tracking**: Comprehensive error handling

#### Pipeline Wrapper

**Location**: `phoenix_observability/instrumentation/pipeline_wrapper.py`

End-to-end pipeline instrumentation:

- **Total latency**: Tracks complete pipeline execution time
- **Pipeline identification**: Names pipelines for easy identification
- **Nested span support**: Works with other instrumentation

### 4. Utility Modules

#### Cost Tracker

**Location**: `phoenix_observability/utils/cost_tracker.py`

- **Model pricing database**: Built-in pricing for major LLM providers
- **Cost calculation**: Accurate cost calculation based on token usage
- **Custom pricing support**: Allows custom pricing models
- **Caching**: LRU cache for pricing lookups

#### Rate Limiter

**Location**: `phoenix_observability/utils/rate_limiter.py`

- **Token bucket algorithm**: Per-second rate limiting
- **Sliding window**: Per-minute rate limiting
- **Per-API limiters**: Separate limiters for different APIs
- **Thread-safe**: Safe for concurrent use

#### HTTP Client Pool

**Location**: `phoenix_observability/utils/http_client.py`

- **Connection pooling**: Reuses HTTP connections
- **Retry strategy**: Automatic retries for transient failures
- **Configurable**: Pool size and timeout settings
- **Thread-safe singleton**: Shared across the application

#### Sanitization

**Location**: `phoenix_observability/utils/sanitize.py`

- **Prompt sanitization**: Truncates long prompts
- **Response sanitization**: Truncates long responses
- **Dictionary sanitization**: Recursive sanitization of nested structures
- **Configurable limits**: All limits configurable via config

#### Security

**Location**: `phoenix_observability/utils/security.py`

- **API key validation**: Validates API key formats
- **URL sanitization**: Validates and sanitizes URLs
- **Name sanitization**: Prevents injection attacks
- **Sensitive data redaction**: Redacts sensitive data from logs

## Data Flow

### LLM Call Instrumentation Flow

```
1. User calls instrumented function
   ↓
2. LLM wrapper creates OpenTelemetry span
   ↓
3. Extracts prompt, model name, and other metadata
   ↓
4. Executes actual LLM call
   ↓
5. Extracts response, usage data, and cost
   ↓
6. Optionally runs PII detection and hallucination detection
   ↓
7. Attaches all data to span as attributes
   ↓
8. Closes span (automatically exported via batch processor)
   ↓
9. Span exported to Phoenix/OTLP endpoint
```

### Span Attribute Structure

Spans contain rich metadata:

- **Basic**: `llm.model_name`, `llm.prompt`, `llm.response`
- **Metrics**: `llm.latency_ms`, `llm.cost`, `llm.input_tokens`, `llm.output_tokens`
- **Evaluation**: `evaluation.hallucination.score`, `evaluation.accuracy.score`
- **Safety**: `safety.pii_detected`, `safety.toxicity_score`
- **Metadata**: `service.name`, `service.version`, `deployment.environment`

## Thread Safety

The package is designed for multi-threaded environments:

- **Configuration**: Thread-safe singleton with double-checked locking
- **Rate limiters**: Thread-safe with locks
- **HTTP client pool**: Thread-safe singleton
- **OpenTelemetry**: Thread-safe by design

## Performance Considerations

### Caching

- **API key caching**: Reduces environment variable reads
- **Pricing lookup caching**: LRU cache for model pricing
- **Configuration caching**: Singleton pattern reduces initialization overhead

### Connection Pooling

- **HTTP connections**: Reused across requests
- **Configurable pool size**: Adjustable for different workloads
- **Automatic retries**: Handles transient failures

### Rate Limiting

- **Token bucket**: Allows bursts while maintaining average rate
- **Per-API limiters**: Prevents one API from affecting others
- **Configurable**: Can be disabled or adjusted per API

### Batch Processing

- **Configurable queue size**: Adjustable for high-throughput scenarios
- **Batch export**: Reduces network overhead
- **Timeout settings**: Balances latency vs. efficiency

## Extensibility

### Adding New Instrumentation

1. Create a new wrapper in `instrumentation/`
2. Use shared helpers from `instrumentation/span_helpers.py`
3. Follow the pattern of existing wrappers
4. Add tests in `tests/`

### Adding New Utilities

1. Create module in `utils/`
2. Add configuration options to `config.py` if needed
3. Export from `__init__.py` if public API
4. Add documentation and tests

### Custom Exporters

The package uses standard OpenTelemetry exporters. To use custom exporters:

1. Configure OpenTelemetry manually
2. Use `get_tracer()` to get tracer instances
3. Create spans manually or use existing wrappers

## Error Handling

### Strategy

- **Fail gracefully**: Errors in instrumentation don't break user code
- **Logging**: Comprehensive logging for debugging
- **Error spans**: Errors attached to spans for observability
- **Sensitive data redaction**: Prevents data leaks in error messages

### Error Types

- **Configuration errors**: Invalid settings
- **Network errors**: OTLP export failures
- **API errors**: External API call failures
- **Validation errors**: Invalid input data

## Security Architecture

### Input Validation

- **URL validation**: Prevents SSRF attacks
- **Name validation**: Prevents injection attacks
- **Type validation**: Ensures correct data types

### Data Protection

- **Sensitive data redaction**: Removes API keys, tokens from logs
- **Secure defaults**: `otlp_insecure=false` by default
- **API key validation**: Validates format before use

## Testing Architecture

### Test Structure

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test OpenTelemetry setup and export
- **Performance benchmarks**: Track performance regressions

### Test Coverage

- Target: 80%+ code coverage
- Focus: Public APIs and critical paths
- Tools: pytest, pytest-cov, pytest-mock

## Future Enhancements

### Planned Features

- **Async/await support**: Full async instrumentation
- **Metrics export**: Prometheus metrics support
- **Custom exporters**: Support for more backends
- **Distributed tracing**: Better support for distributed systems

### Performance Improvements

- **Async I/O**: Non-blocking API calls
- **Batch optimization**: Smarter batching strategies
- **Compression**: Compress span data for export

## Conclusion

Phoenix Observability is designed to be:

- **Lightweight**: Minimal overhead on application performance
- **Configurable**: Extensive configuration options
- **Extensible**: Easy to add new features
- **Reliable**: Comprehensive error handling and testing
- **Secure**: Built-in security best practices

For questions or contributions, see the [Contributing Guide](../README.md#contributing).

