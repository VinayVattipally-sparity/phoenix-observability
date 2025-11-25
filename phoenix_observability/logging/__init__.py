"""Logging utilities for span enrichment."""

from phoenix_observability.logging.span_utils import set_nested_attribute, set_span_attributes
from phoenix_observability.logging.enrich import enrich_span_with_context, enrich_current_span

__all__ = [
    "set_nested_attribute",
    "set_span_attributes",
    "enrich_span_with_context",
    "enrich_current_span",
]

