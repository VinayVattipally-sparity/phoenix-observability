"""
Performance benchmarks for phoenix-observability.

Tests key operations for performance regressions.
"""

import time
import pytest
from unittest.mock import Mock, patch

from phoenix_observability.utils.sanitize import sanitize_prompt, sanitize_response, sanitize_dict
from phoenix_observability.utils.cost_tracker import calculate_cost
from phoenix_observability.utils.rate_limiter import get_rate_limiter_manager
from phoenix_observability.utils.http_client import get_http_client
from phoenix_observability.config import get_config


class TestSanitizationPerformance:
    """Benchmark sanitization operations."""
    
    def test_sanitize_prompt_performance(self, benchmark):
        """Benchmark prompt sanitization."""
        long_prompt = "This is a test prompt. " * 1000  # ~25KB
        
        result = benchmark(sanitize_prompt, long_prompt)
        assert isinstance(result, str)
        assert len(result) <= get_config().max_prompt_length
    
    def test_sanitize_response_performance(self, benchmark):
        """Benchmark response sanitization."""
        long_response = "This is a test response. " * 2000  # ~50KB
        
        result = benchmark(sanitize_response, long_response)
        assert isinstance(result, str)
        assert len(result) <= get_config().max_response_length
    
    def test_sanitize_dict_performance(self, benchmark):
        """Benchmark dictionary sanitization."""
        large_dict = {
            f"key_{i}": f"value_{i}" * 100
            for i in range(100)
        }
        
        result = benchmark(sanitize_dict, large_dict)
        assert isinstance(result, dict)
        assert len(result) == 100


class TestCostCalculationPerformance:
    """Benchmark cost calculation operations."""
    
    def test_calculate_cost_performance(self, benchmark):
        """Benchmark cost calculation."""
        result = benchmark(
            calculate_cost,
            model_name="gpt-4",
            input_tokens=1000,
            output_tokens=500
        )
        assert isinstance(result, float)
        assert result >= 0


class TestRateLimiterPerformance:
    """Benchmark rate limiter operations."""
    
    def test_rate_limiter_acquire_performance(self, benchmark):
        """Benchmark rate limiter acquire operation."""
        manager = get_rate_limiter_manager()
        limiter = manager.get_limiter("test_api", requests_per_second=1000)
        
        def acquire():
            return limiter.acquire()
        
        result = benchmark(acquire)
        assert isinstance(result, bool)
    
    def test_rate_limiter_concurrent_performance(self):
        """Test rate limiter under concurrent load."""
        import threading
        
        manager = get_rate_limiter_manager()
        limiter = manager.get_limiter("concurrent_test", requests_per_second=100)
        
        results = []
        
        def worker():
            for _ in range(10):
                results.append(limiter.acquire())
                time.sleep(0.001)  # Small delay to simulate work
        
        threads = [threading.Thread(target=worker) for _ in range(10)]
        start_time = time.time()
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        elapsed = time.time() - start_time
        
        # Should allow most requests (rate limit is 100/sec)
        # With 10 threads * 10 requests = 100 requests
        # Should complete in roughly 1 second
        assert elapsed < 2.0  # Allow some overhead
        assert sum(results) > 50  # At least half should succeed


class TestHTTPClientPerformance:
    """Benchmark HTTP client operations."""
    
    @pytest.mark.skipif(
        not hasattr(pytest, 'benchmark'),
        reason="pytest-benchmark not installed"
    )
    def test_http_client_creation_performance(self, benchmark):
        """Benchmark HTTP client pool creation."""
        # This is mainly to ensure creation is fast
        def create_client():
            from phoenix_observability.utils.http_client import HTTPClientPool
            from phoenix_observability.config import get_config
            config = get_config()
            return HTTPClientPool(
                pool_connections=config.http_pool_connections,
                pool_maxsize=config.http_pool_maxsize
            )
        
        client = benchmark(create_client)
        assert client is not None
        client.close()


class TestConfigPerformance:
    """Benchmark configuration operations."""
    
    def test_get_config_performance(self, benchmark):
        """Benchmark config retrieval."""
        result = benchmark(get_config)
        assert result is not None


@pytest.fixture
def benchmark():
    """
    Simple benchmark fixture.
    
    If pytest-benchmark is available, use it. Otherwise, use simple timing.
    """
    try:
        import pytest_benchmark
        # pytest-benchmark will provide the fixture automatically
        return pytest_benchmark.fixture.benchmark
    except ImportError:
        # Fallback to simple timing
        def simple_benchmark(func, *args, **kwargs):
            iterations = 100
            start = time.time()
            for _ in range(iterations):
                result = func(*args, **kwargs)
            elapsed = time.time() - start
            avg_time = elapsed / iterations
            print(f"\n{func.__name__}: {avg_time*1000:.2f}ms per call (avg over {iterations} iterations)")
            return result
        return simple_benchmark

