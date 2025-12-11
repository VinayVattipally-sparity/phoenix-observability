# Configuration Guide

Complete guide to configuring Phoenix Observability.

## Overview

Phoenix Observability is highly configurable through environment variables. All settings can be set in a `.env` file or as system environment variables.

## Quick Configuration

Create a `.env` file in your project root:

```env
# Required
PHOENIX_ENDPOINT=https://phoenix.example.com

# Recommended
SERVICE_NAME=my-llm-service
ENVIRONMENT=production
```

## Configuration Options

### Core Settings

#### `PHOENIX_ENDPOINT` (Required)
Phoenix server endpoint URL.

- **Type**: String (URL)
- **Default**: `https://sparity-phoenix.com`
- **Example**: `https://phoenix.example.com`
- **Note**: Must be a valid HTTP/HTTPS URL or gRPC endpoint (`grpc://host:port`)

#### `SERVICE_NAME`
Default service name for all spans.

- **Type**: String
- **Default**: `phoenix_observability`
- **Example**: `my-llm-service`
- **Note**: Can be overridden per-instrumentation with `service_name` parameter

#### `ENVIRONMENT`
Deployment environment identifier.

- **Type**: String
- **Default**: `dev`
- **Example**: `production`, `staging`, `dev`
- **Note**: Used in resource attributes for filtering

### Feature Flags

#### `ENABLE_COST_TRACKING`
Enable automatic cost calculation for LLM calls.

- **Type**: Boolean
- **Default**: `true`
- **Example**: `true`, `false`
- **Note**: Can be overridden per-instrumentation with `track_cost` parameter

#### `ENABLE_PII_TRACKING`
Enable PII detection and safety analysis.

- **Type**: Boolean
- **Default**: `true`
- **Example**: `true`, `false`
- **Note**: Can be overridden per-instrumentation with `track_pii` parameter

#### `ENABLE_GPU_TRACKING`
Enable GPU usage monitoring.

- **Type**: Boolean
- **Default**: `false`
- **Example**: `true`, `false`
- **Note**: Requires `pynvml` package

### OTLP Settings

#### `OTLP_ENDPOINT`
Custom OTLP endpoint (overrides Phoenix endpoint).

- **Type**: String (URL)
- **Default**: `{PHOENIX_ENDPOINT}/v1/traces`
- **Example**: `https://otel-collector.example.com:4318/v1/traces`
- **Note**: For HTTP, must include `/v1/traces` path

#### `OTLP_INSECURE`
Allow insecure connections (TLS disabled).

- **Type**: Boolean
- **Default**: `false`
- **Example**: `true`, `false`
- **Security**: Only use `true` for local development. Production should always use `false`.

#### `OTEL_EXPORTER_OTLP_HEADERS`
Custom headers for OTLP export.

- **Type**: String (comma-separated key=value pairs)
- **Default**: None
- **Example**: `Authorization=Bearer token,User-Agent=my-app`
- **Note**: Used for authentication with OTLP endpoints

### Batch Processing

#### `BATCH_TIMEOUT_MS`
Timeout for batch export in milliseconds.

- **Type**: Integer
- **Default**: `5000`
- **Example**: `10000`
- **Note**: Higher values reduce network calls but increase latency

#### `MAX_EXPORT_BATCH_SIZE`
Maximum number of spans per batch.

- **Type**: Integer
- **Default**: `512`
- **Example**: `1024`
- **Note**: Larger batches are more efficient but use more memory

#### `MAX_QUEUE_SIZE`
Maximum number of spans in export queue.

- **Type**: Integer
- **Default**: `2048`
- **Example**: `4096`
- **Note**: Increase for high-throughput scenarios

### Sanitization Limits

#### `MAX_PROMPT_LENGTH`
Maximum length for prompts before truncation.

- **Type**: Integer
- **Default**: `10000`
- **Example**: `20000`
- **Note**: Longer prompts use more storage

#### `MAX_RESPONSE_LENGTH`
Maximum length for responses before truncation.

- **Type**: Integer
- **Default**: `50000`
- **Example**: `100000`
- **Note**: Longer responses use more storage

#### `MAX_CONTEXT_LENGTH`
Maximum length for RAG context before truncation.

- **Type**: Integer
- **Default**: `50000`
- **Example**: `100000`
- **Note**: Used for RAG document contexts

#### `MAX_VALUE_LENGTH`
Maximum length for dictionary values before truncation.

- **Type**: Integer
- **Default**: `1000`
- **Example**: `2000`
- **Note**: Used in sanitize_dict()

#### `MAX_SPAN_ATTRIBUTE_LENGTH`
Maximum length for span attribute values.

- **Type**: Integer
- **Default**: `1000`
- **Example**: `2000`
- **Note**: OpenTelemetry attribute length limit

### Rate Limiting

#### `RATE_LIMIT_ENABLED`
Enable rate limiting for external API calls.

- **Type**: Boolean
- **Default**: `true`
- **Example**: `true`, `false`
- **Note**: Prevents API abuse

#### `RATE_LIMIT_REQUESTS_PER_SECOND`
Maximum requests per second per API.

- **Type**: Integer
- **Default**: `10`
- **Example**: `20`
- **Note**: Token bucket algorithm

#### `RATE_LIMIT_REQUESTS_PER_MINUTE`
Maximum requests per minute per API.

- **Type**: Integer
- **Default**: `60`
- **Example**: `120`
- **Note**: Sliding window algorithm

### HTTP Connection Pooling

#### `HTTP_POOL_CONNECTIONS`
Number of connection pools to cache.

- **Type**: Integer
- **Default**: `10`
- **Example**: `20`
- **Note**: More pools = better concurrency

#### `HTTP_POOL_MAXSIZE`
Maximum connections per pool.

- **Type**: Integer
- **Default**: `20`
- **Example**: `50`
- **Note**: More connections = better throughput

#### `HTTP_TIMEOUT`
Request timeout in seconds.

- **Type**: Integer
- **Default**: `30`
- **Example**: `60`
- **Note**: Timeout for HTTP requests

### Toxicity Detection

#### `TOXICITY_DETECTION_METHOD`
Method for toxicity detection.

- **Type**: String
- **Default**: `auto`
- **Options**: `auto`, `openai`, `perspective`, `heuristic`
- **Note**: `auto` selects best available method

## Programmatic Configuration

You can also configure programmatically:

```python
from phoenix_observability.config import ObservabilityConfig
import os

# Set environment variables
os.environ["PHOENIX_ENDPOINT"] = "https://phoenix.example.com"
os.environ["SERVICE_NAME"] = "my-service"
os.environ["MAX_PROMPT_LENGTH"] = "20000"

# Access config
config = ObservabilityConfig()
print(config.phoenix_endpoint)
print(config.max_prompt_length)
```

## Configuration Priority

1. Function parameters (highest priority)
2. Environment variables
3. Config defaults (lowest priority)

## Example Configuration Files

### Development

```env
PHOENIX_ENDPOINT=http://localhost:6006
ENVIRONMENT=dev
SERVICE_NAME=dev-service
OTLP_INSECURE=true
ENABLE_GPU_TRACKING=false
```

### Production

```env
PHOENIX_ENDPOINT=https://phoenix.production.com
ENVIRONMENT=production
SERVICE_NAME=production-service
OTLP_INSECURE=false
ENABLE_COST_TRACKING=true
ENABLE_PII_TRACKING=true
MAX_QUEUE_SIZE=4096
RATE_LIMIT_REQUESTS_PER_SECOND=20
```

### High-Throughput

```env
PHOENIX_ENDPOINT=https://phoenix.example.com
SERVICE_NAME=high-throughput-service
MAX_QUEUE_SIZE=8192
MAX_EXPORT_BATCH_SIZE=1024
BATCH_TIMEOUT_MS=10000
HTTP_POOL_CONNECTIONS=20
HTTP_POOL_MAXSIZE=50
```

## Validation

Configuration values are validated on initialization. Invalid values will raise errors or fall back to defaults with warnings.

## Next Steps

- See [Quick Start](quick-start.md) for basic setup
- Check [Examples](../examples.md) for usage patterns
- Review [API Reference](../api/core/config.md) for detailed API docs

