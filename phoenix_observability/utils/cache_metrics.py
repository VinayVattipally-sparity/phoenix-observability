"""
Cache metrics tracking utilities.

Tracks cache hits, misses, and hit rates.
"""

import logging
from typing import Dict, Optional

from opentelemetry import trace

logger = logging.getLogger(__name__)


def attach_cache_metrics_to_span(
    span: trace.Span,
    hits: int = 0,
    misses: int = 0,
    size: int = 0,
) -> None:
    """
    Attach cache metrics to a span.
    
    Args:
        span: OpenTelemetry span
        hits: Number of cache hits
        misses: Number of cache misses
        size: Current cache size
    """
    try:
        span.set_attribute("cache.hits", hits)
        span.set_attribute("cache.misses", misses)
        span.set_attribute("cache.size", size)
        
        # Calculate hit rate
        total_requests = hits + misses
        if total_requests > 0:
            hit_rate_percent = (hits / total_requests) * 100.0
            span.set_attribute("cache.hit_rate_percent", round(hit_rate_percent, 2))
        else:
            span.set_attribute("cache.hit_rate_percent", 0.0)
    except Exception as e:
        logger.debug(f"Failed to attach cache metrics: {e}")

