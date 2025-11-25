"""
RAG instrumentation wrapper.

Wraps vector search and retrievers to log documents, latency, and metadata.
"""

import functools
import logging
from typing import Any, Callable, List, Optional

from opentelemetry import trace

from phoenix_observability.otel_setup import get_tracer
from phoenix_observability.utils.latency import LatencyTimer
from phoenix_observability.instrumentation.error_handler import handle_error

logger = logging.getLogger(__name__)


def instrument_retriever(
    retriever_name: Optional[str] = None,
    log_documents: bool = True,
    log_metadata: bool = True,
):
    """
    Decorator to instrument RAG retrieval operations.

    Args:
        retriever_name: Name of the retriever (defaults to function name)
        log_documents: Whether to log document contents
        log_metadata: Whether to log metadata

    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = retriever_name or func.__name__

            # Create span
            with tracer.start_as_current_span(f"rag.retrieve.{name}") as span:
                # Set OpenInference span kind for retrievers
                span.set_attribute("openinference.span.kind", "RETRIEVER")
                
                span.set_attribute("rag.retriever", name)
                span.set_attribute("rag.function", func.__name__)

                # Extract query if available
                query = None
                if args and len(args) > 0:
                    query = args[0]
                elif "query" in kwargs:
                    query = kwargs["query"]

                if query:
                    query_str = str(query)[:1000]  # Truncate long queries
                    span.set_attribute("rag.query", query_str)

                timer = LatencyTimer()
                timer.start()

                try:
                    # Call the retriever function
                    result = func(*args, **kwargs)

                    # Measure latency
                    latency = timer.stop()
                    latency_ms = latency * 1000  # Convert to milliseconds
                    span.set_attribute("rag.retriever.latency_ms", latency_ms)
                    # Keep old name for backward compatibility
                    span.set_attribute("rag.latency_seconds", latency)

                    # Extract documents
                    documents = result
                    if isinstance(result, dict):
                        # Common formats
                        if "documents" in result:
                            documents = result["documents"]
                        elif "docs" in result:
                            documents = result["docs"]
                        elif "results" in result:
                            documents = result["results"]

                    # Count documents
                    if isinstance(documents, list):
                        num_docs = len(documents)
                        span.set_attribute("rag.documents_returned", num_docs)
                        # Keep old name for backward compatibility
                        span.set_attribute("rag.documents_count", num_docs)
                        
                        # Build context string for hallucination detection
                        context_parts = []
                        for doc in documents[:10]:  # Limit to first 10 for context
                            doc_content = str(doc)
                            if hasattr(doc, "page_content"):
                                doc_content = doc.page_content
                            elif isinstance(doc, dict):
                                doc_content = doc.get("content", doc.get("text", str(doc)))
                            context_parts.append(doc_content)
                        
                        # Set rag.context for Phoenix hallucination detection
                        if context_parts:
                            context_str = "\n\n".join(context_parts)
                            # Truncate to reasonable length (Phoenix can handle up to ~50k chars)
                            max_context_length = 50000
                            if len(context_str) > max_context_length:
                                context_str = context_str[:max_context_length] + "... [truncated]"
                            span.set_attribute("rag.context", context_str)

                        # Log document contents if enabled
                        if log_documents and num_docs > 0:
                            # Log first few documents (truncated)
                            for i, doc in enumerate(documents[:5]):  # Limit to first 5
                                doc_content = str(doc)
                                if hasattr(doc, "page_content"):
                                    doc_content = doc.page_content
                                elif isinstance(doc, dict):
                                    doc_content = doc.get("content", doc.get("text", str(doc)))

                                # Truncate long documents
                                truncated = doc_content[:500] if len(doc_content) > 500 else doc_content
                                span.set_attribute(f"rag.document.{i}.content", truncated)

                                # Log metadata if available
                                if log_metadata:
                                    if hasattr(doc, "metadata") and doc.metadata:
                                        for key, value in list(doc.metadata.items())[:5]:  # Limit metadata keys
                                            span.set_attribute(
                                                f"rag.document.{i}.metadata.{key}",
                                                str(value)[:200],
                                            )
                                    elif isinstance(doc, dict) and "metadata" in doc:
                                        for key, value in list(doc["metadata"].items())[:5]:
                                            span.set_attribute(
                                                f"rag.document.{i}.metadata.{key}",
                                                str(value)[:200],
                                            )

                    elif isinstance(documents, dict):
                        # Handle dict results
                        span.set_attribute("rag.result_type", "dict")
                        if "count" in documents:
                            span.set_attribute("rag.documents_returned", documents["count"])
                            # Keep old name for backward compatibility
                            span.set_attribute("rag.documents_count", documents["count"])

                    span.set_status(trace.Status(trace.StatusCode.OK))
                    return result

                except Exception as e:
                    timer.stop()
                    handle_error(span, e, {"retriever": name, "function": func.__name__})
                    raise

        return wrapper

    return decorator

