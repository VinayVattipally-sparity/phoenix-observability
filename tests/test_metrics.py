"""
Unit tests for metrics.py module.
"""

import pytest
from phoenix_observability.utils.metrics import (
    get_metrics_collector,
    reset_metrics,
    record_span_created,
    record_span_latency,
    record_cost_calculated,
    record_cost_amount,
    record_error,
    MetricsCollector,
    MetricCounter,
    MetricHistogram,
)


class TestMetricCounter:
    """Tests for MetricCounter class."""

    def test_counter_increment(self):
        """Test counter increment."""
        counter = MetricCounter("test_counter")
        assert counter.get() == 0
        counter.increment()
        assert counter.get() == 1
        counter.increment(5)
        assert counter.get() == 6

    def test_counter_reset(self):
        """Test counter reset."""
        counter = MetricCounter("test_counter")
        counter.increment(10)
        assert counter.get() == 10
        counter.reset()
        assert counter.get() == 0

    def test_counter_thread_safety(self):
        """Test counter thread safety."""
        import threading
        
        counter = MetricCounter("test_counter")
        
        def increment_many():
            for _ in range(100):
                counter.increment()
        
        threads = [threading.Thread(target=increment_many) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        
        assert counter.get() == 1000


class TestMetricHistogram:
    """Tests for MetricHistogram class."""

    def test_histogram_record(self):
        """Test histogram recording."""
        histogram = MetricHistogram("test_histogram")
        histogram.record(10.0)
        histogram.record(20.0)
        histogram.record(30.0)
        
        stats = histogram.get_stats()
        assert stats["count"] == 3
        assert stats["min"] == 10.0
        assert stats["max"] == 30.0
        assert stats["avg"] == 20.0

    def test_histogram_empty(self):
        """Test histogram with no values."""
        histogram = MetricHistogram("test_histogram")
        stats = histogram.get_stats()
        assert stats["count"] == 0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["avg"] == 0.0

    def test_histogram_reset(self):
        """Test histogram reset."""
        histogram = MetricHistogram("test_histogram")
        histogram.record(10.0)
        histogram.reset()
        stats = histogram.get_stats()
        assert stats["count"] == 0


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_get_counter(self):
        """Test getting or creating counter."""
        collector = MetricsCollector()
        counter1 = collector.get_counter("test")
        counter2 = collector.get_counter("test")
        assert counter1 is counter2

    def test_get_histogram(self):
        """Test getting or creating histogram."""
        collector = MetricsCollector()
        hist1 = collector.get_histogram("test")
        hist2 = collector.get_histogram("test")
        assert hist1 is hist2

    def test_increment_span_created(self):
        """Test span creation tracking."""
        collector = MetricsCollector()
        collector.increment_span_created("llm")
        collector.increment_span_created("rag")
        
        metrics = collector.get_all_metrics()
        assert metrics["counters"]["spans.created.llm"] == 1
        assert metrics["counters"]["spans.created.rag"] == 1
        assert metrics["counters"]["spans.created.total"] == 2

    def test_record_span_latency(self):
        """Test span latency tracking."""
        collector = MetricsCollector()
        collector.record_span_latency(100.0, "llm")
        collector.record_span_latency(200.0, "llm")
        
        metrics = collector.get_all_metrics()
        llm_stats = metrics["histograms"]["spans.latency.llm"]
        assert llm_stats["count"] == 2
        assert llm_stats["min"] == 100.0
        assert llm_stats["max"] == 200.0

    def test_increment_cost_calculated(self):
        """Test cost calculation tracking."""
        collector = MetricsCollector()
        collector.increment_cost_calculated("gpt-4")
        collector.increment_cost_calculated("gpt-4")
        collector.increment_cost_calculated("claude-3")
        
        metrics = collector.get_all_metrics()
        assert metrics["counters"]["cost.calculated.gpt-4"] == 2
        assert metrics["counters"]["cost.calculated.claude-3"] == 1
        assert metrics["counters"]["cost.calculated.total"] == 3

    def test_record_cost_amount(self):
        """Test cost amount tracking."""
        collector = MetricsCollector()
        collector.record_cost_amount(10.5, "gpt-4")
        collector.record_cost_amount(20.0, "gpt-4")
        
        metrics = collector.get_all_metrics()
        gpt4_stats = metrics["histograms"]["cost.amount.gpt-4"]
        assert gpt4_stats["count"] == 2
        assert gpt4_stats["avg"] == 15.25

    def test_increment_error(self):
        """Test error tracking."""
        collector = MetricsCollector()
        collector.increment_error("ValueError")
        collector.increment_error("TypeError")
        collector.increment_error("ValueError")
        
        metrics = collector.get_all_metrics()
        assert metrics["counters"]["errors.ValueError"] == 2
        assert metrics["counters"]["errors.TypeError"] == 1
        assert metrics["counters"]["errors.total"] == 3

    def test_reset_all(self):
        """Test resetting all metrics."""
        collector = MetricsCollector()
        collector.increment_span_created("llm")
        collector.record_span_latency(100.0, "llm")
        
        collector.reset_all()
        metrics = collector.get_all_metrics()
        assert metrics["counters"] == {}
        assert metrics["histograms"] == {}


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_record_span_created(self):
        """Test record_span_created convenience function."""
        reset_metrics()
        record_span_created("llm")
        collector = get_metrics_collector()
        metrics = collector.get_all_metrics()
        assert metrics["counters"]["spans.created.llm"] == 1

    def test_record_span_latency(self):
        """Test record_span_latency convenience function."""
        reset_metrics()
        record_span_latency(150.0, "llm")
        collector = get_metrics_collector()
        metrics = collector.get_all_metrics()
        llm_stats = metrics["histograms"]["spans.latency.llm"]
        assert llm_stats["count"] == 1
        assert llm_stats["avg"] == 150.0

    def test_record_cost_calculated(self):
        """Test record_cost_calculated convenience function."""
        reset_metrics()
        record_cost_calculated("gpt-4")
        collector = get_metrics_collector()
        metrics = collector.get_all_metrics()
        assert metrics["counters"]["cost.calculated.gpt-4"] == 1

    def test_record_cost_amount(self):
        """Test record_cost_amount convenience function."""
        reset_metrics()
        record_cost_amount(25.5, "gpt-4")
        collector = get_metrics_collector()
        metrics = collector.get_all_metrics()
        gpt4_stats = metrics["histograms"]["cost.amount.gpt-4"]
        assert gpt4_stats["count"] == 1
        assert gpt4_stats["avg"] == 25.5

    def test_record_error(self):
        """Test record_error convenience function."""
        reset_metrics()
        record_error("ValueError")
        collector = get_metrics_collector()
        metrics = collector.get_all_metrics()
        assert metrics["counters"]["errors.ValueError"] == 1

