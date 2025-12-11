# Quick Start Guide

Get started with Phoenix Observability in minutes.

## Prerequisites

- Python 3.9 or higher
- A Phoenix server endpoint (or use the default)

## Installation

```bash
pip install phoenix-observability
```

For Phoenix UI support:

```bash
pip install phoenix-observability[phoenix]
```

## Basic Setup

### 1. Create a `.env` file

Create a `.env` file in your project root:

```env
PHOENIX_ENDPOINT=https://phoenix-sparity.com
ENVIRONMENT=dev
SERVICE_NAME=my-llm-service
```

### 2. Initialize Observability

```python
from phoenix_observability import init_observability

# Initialize with service name
init_observability(service_name="my-llm-service")
```

### 3. Instrument Your Code

#### LLM Calls

```python
from phoenix_observability import instrument_llm

@instrument_llm(model_name="gpt-4")
def call_openai(prompt: str):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

#### RAG Systems

```python
from phoenix_observability import instrument_retriever

@instrument_retriever
def retrieve_documents(query: str):
    # Your retrieval logic
    return documents
```

#### Agents

```python
from phoenix_observability import instrument_agent

@instrument_agent
def my_agent(input_data):
    # Your agent logic
    return result
```

## Next Steps

- Read the [Configuration Guide](configuration.md) for advanced settings
- Check out [Usage Examples](../examples.md) for more patterns
- Explore the [API Reference](../api/core/config.md) for detailed documentation

