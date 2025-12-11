# Usage Examples

Comprehensive examples for using Phoenix Observability in various scenarios.

## Table of Contents

- [Basic LLM Instrumentation](#basic-llm-instrumentation)
- [Advanced LLM Instrumentation](#advanced-llm-instrumentation)
- [RAG System Instrumentation](#rag-system-instrumentation)
- [Agent Instrumentation](#agent-instrumentation)
- [Pipeline Instrumentation](#pipeline-instrumentation)
- [Cost Tracking](#cost-tracking)
- [Error Handling](#error-handling)
- [Custom Configuration](#custom-configuration)

## Basic LLM Instrumentation

### Simple OpenAI Call

```python
from phoenix_observability import init_observability, instrument_llm
from openai import OpenAI

init_observability(service_name="chatbot")

@instrument_llm
def chat(prompt: str):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Use it
result = chat("What is the capital of France?")
```

### With Model Name

```python
@instrument_llm(model_name="gpt-4")
def chat_with_model(prompt: str):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

## Advanced LLM Instrumentation

### With Cost Tracking

```python
@instrument_llm(track_cost=True)
def expensive_llm_call(prompt: str):
    # Cost will be automatically calculated and attached to the span
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    return response.choices[0].message.content
```

### With PII Detection

```python
@instrument_llm(track_pii=True)
def sensitive_llm_call(prompt: str):
    # PII detection will run automatically
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

### With Structured Output Validation

```python
@instrument_llm(
    expected_schema={
        "name": str,
        "age": int,
        "email": str
    }
)
def extract_user_info(text: str):
    # Response will be validated against the schema
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Extract user information as JSON"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content
```

## RAG System Instrumentation

### Basic Retrieval

```python
from phoenix_observability import instrument_retriever

@instrument_retriever
def retrieve_documents(query: str):
    # Your vector search or retrieval logic
    from your_rag_system import search
    results = search(query, top_k=5)
    return results
```

### With Document Logging

```python
@instrument_retriever(log_documents=True, log_metadata=True)
def retrieve_with_logging(query: str):
    # Documents and metadata will be logged to spans
    results = your_vector_db.search(query)
    return results
```

### RAG Pipeline

```python
from phoenix_observability import instrument_retriever, instrument_llm

@instrument_retriever
def retrieve(query: str):
    return vector_db.search(query)

@instrument_llm
def generate(context: str, query: str):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Context: {context}"},
            {"role": "user", "content": query}
        ]
    )
    return response.choices[0].message.content

# Use together
def rag_pipeline(query: str):
    docs = retrieve(query)
    context = "\n".join([doc.content for doc in docs])
    return generate(context, query)
```

## Agent Instrumentation

### Basic Agent

```python
from phoenix_observability import instrument_agent

@instrument_agent
def my_agent(user_input: str):
    # Your agent logic
    tools = get_available_tools()
    result = agent.run(user_input, tools=tools)
    return result
```

### With Tool Logging

```python
@instrument_agent(
    log_tool_inputs=True,
    log_tool_outputs=True,
    log_intermediate_steps=True
)
def detailed_agent(user_input: str):
    # All tool calls and intermediate steps will be logged
    result = agent.run(user_input)
    return result
```

## Pipeline Instrumentation

### End-to-End Pipeline

```python
from phoenix_observability import instrument_pipeline

@instrument_pipeline(pipeline_name="document-processing")
def process_document(doc: str):
    # This tracks end-to-end latency
    cleaned = clean_text(doc)
    analyzed = analyze(cleaned)
    summarized = summarize(analyzed)
    return summarized
```

### Multi-Stage Pipeline

```python
@instrument_pipeline(pipeline_name="rag-pipeline")
def complete_rag_pipeline(query: str):
    # Tracks total pipeline latency
    docs = retrieve_documents(query)
    context = format_context(docs)
    response = generate_response(context, query)
    return response
```

## Cost Tracking

### Manual Cost Calculation

```python
from phoenix_observability.utils.cost_tracker import calculate_cost

cost = calculate_cost(
    model_name="gpt-4",
    input_tokens=1000,
    output_tokens=500
)
print(f"Cost: ${cost:.4f}")
```

### With Custom Pricing

```python
cost = calculate_cost(
    model_name="custom-model",
    input_tokens=1000,
    output_tokens=500,
    custom_pricing={
        "input": 0.001,  # $0.001 per input token
        "output": 0.002  # $0.002 per output token
    }
)
```

## Error Handling

### Automatic Error Tracking

```python
@instrument_llm
def risky_llm_call(prompt: str):
    try:
        # Your LLM call
        return result
    except Exception as e:
        # Error will be automatically attached to the span
        raise
```

### Custom Error Handling

```python
from phoenix_observability.instrumentation.error_handler import handle_error

@instrument_llm
def custom_error_handling(prompt: str):
    try:
        return llm_call(prompt)
    except Exception as e:
        handle_error(e, context={"prompt": prompt})
        raise
```

## Custom Configuration

### Environment Variables

```env
# .env file
PHOENIX_ENDPOINT=https://phoenix.example.com
SERVICE_NAME=my-service
ENVIRONMENT=production
ENABLE_COST_TRACKING=true
ENABLE_PII_TRACKING=true
MAX_PROMPT_LENGTH=20000
RATE_LIMIT_REQUESTS_PER_SECOND=20
```

### Programmatic Configuration

```python
from phoenix_observability.config import ObservabilityConfig
import os

# Set environment variables programmatically
os.environ["PHOENIX_ENDPOINT"] = "https://phoenix.example.com"
os.environ["SERVICE_NAME"] = "my-service"

# Or use config directly
config = ObservabilityConfig()
print(config.phoenix_endpoint)
print(config.max_prompt_length)
```

## Rate Limiting

### Using Rate Limiter

```python
from phoenix_observability.utils.rate_limiter import get_rate_limiter_manager

manager = get_rate_limiter_manager()
limiter = manager.get_limiter("openai", requests_per_second=10)

# Before making API call
if limiter.acquire():
    # Make your API call
    response = openai_call()
else:
    # Wait and retry
    limiter.wait()
    response = openai_call()
```

## HTTP Connection Pooling

### Using HTTP Client Pool

```python
from phoenix_observability.utils.http_client import get_http_client

http_client = get_http_client()

# Use pooled connections
response = http_client.post(
    "https://api.example.com/endpoint",
    json={"data": "value"}
)
```

## Next Steps

- Read the [API Reference](../api/core/config.md) for detailed documentation
- Check out [Architecture](../architecture.md) for system design
- Review [Security](../security.md) for best practices

