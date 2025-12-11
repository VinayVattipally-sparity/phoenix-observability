"""
Rate limiting utility for external API calls.

Provides token bucket and sliding window rate limiters to prevent
API abuse and respect rate limits.
"""

import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict

from phoenix_observability.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: int = 10
    requests_per_minute: int = 60
    burst_size: int = 5  # Allow small bursts


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.
    
    Allows bursts up to burst_size, then enforces steady rate.
    """
    
    def __init__(
        self,
        requests_per_second: int,
        burst_size: Optional[int] = None,
        name: str = "default"
    ):
        """
        Initialize token bucket rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst size (defaults to requests_per_second / 2)
            name: Name for this rate limiter (for logging)
        """
        self.requests_per_second = max(1, requests_per_second)
        self.burst_size = burst_size or max(1, self.requests_per_second // 2)
        self.tokens = float(self.burst_size)
        self.last_refill = time.time()
        self.name = name
        self.lock = threading.Lock()
        self._total_requests = 0
        self._blocked_requests = 0
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens (make a request).
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill tokens based on elapsed time
            refill_amount = elapsed * self.requests_per_second
            self.tokens = min(self.burst_size, self.tokens + refill_amount)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                self._total_requests += 1
                return True
            else:
                self._blocked_requests += 1
                return False
    
    def wait(self, tokens: int = 1) -> float:
        """
        Wait until tokens are available, then acquire them.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
            
        Returns:
            Wait time in seconds
        """
        wait_time = 0.0
        while not self.acquire(tokens):
            # Calculate how long to wait
            with self.lock:
                needed = tokens - self.tokens
                wait_time = needed / self.requests_per_second
                if wait_time < 0.01:  # Minimum wait
                    wait_time = 0.01
            
            time.sleep(wait_time)
            wait_time += wait_time
        
        return wait_time
    
    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics."""
        with self.lock:
            return {
                "total_requests": self._total_requests,
                "blocked_requests": self._blocked_requests,
                "current_tokens": int(self.tokens),
            }


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.
    
    Tracks requests in a time window and enforces limits.
    """
    
    def __init__(
        self,
        requests_per_minute: int,
        window_seconds: int = 60,
        name: str = "default"
    ):
        """
        Initialize sliding window rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            window_seconds: Time window in seconds (default: 60)
            name: Name for this rate limiter (for logging)
        """
        self.requests_per_minute = max(1, requests_per_minute)
        self.window_seconds = window_seconds
        self.name = name
        self.requests: deque = deque()
        self.lock = threading.Lock()
        self._total_requests = 0
        self._blocked_requests = 0
    
    def acquire(self) -> bool:
        """
        Try to make a request.
        
        Returns:
            True if request is allowed, False if rate limited
        """
        with self.lock:
            now = time.time()
            
            # Remove requests outside the window
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            
            # Check if we're within the limit
            if len(self.requests) < self.requests_per_minute:
                self.requests.append(now)
                self._total_requests += 1
                return True
            else:
                self._blocked_requests += 1
                return False
    
    def wait(self) -> float:
        """
        Wait until a request slot is available, then make the request.
        
        Returns:
            Wait time in seconds
        """
        wait_time = 0.0
        while not self.acquire():
            # Calculate how long to wait
            with self.lock:
                if self.requests:
                    oldest_request = self.requests[0]
                    wait_time = (oldest_request + self.window_seconds) - time.time()
                    if wait_time < 0.01:
                        wait_time = 0.01
                else:
                    wait_time = 0.01
            
            time.sleep(wait_time)
            wait_time += wait_time
        
        return wait_time
    
    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics."""
        with self.lock:
            return {
                "total_requests": self._total_requests,
                "blocked_requests": self._blocked_requests,
                "current_window_requests": len(self.requests),
            }


class RateLimiterManager:
    """
    Manager for multiple rate limiters.
    
    Provides per-API rate limiting with global defaults.
    """
    
    def __init__(self):
        """Initialize rate limiter manager."""
        self.config = get_config()
        self.limiters: Dict[str, TokenBucketRateLimiter] = {}
        self.window_limiters: Dict[str, SlidingWindowRateLimiter] = {}
        self.lock = threading.Lock()
        
        # Create default limiters if rate limiting is enabled
        if self.config.rate_limit_enabled:
            self._create_default_limiters()
    
    def _create_default_limiters(self):
        """Create default rate limiters from config."""
        default_limiter = TokenBucketRateLimiter(
            requests_per_second=self.config.rate_limit_requests_per_second,
            name="default"
        )
        default_window = SlidingWindowRateLimiter(
            requests_per_minute=self.config.rate_limit_requests_per_minute,
            name="default_window"
        )
        
        with self.lock:
            self.limiters["default"] = default_limiter
            self.window_limiters["default"] = default_window
    
    def get_limiter(
        self,
        api_name: str,
        requests_per_second: Optional[int] = None,
        requests_per_minute: Optional[int] = None
    ) -> TokenBucketRateLimiter:
        """
        Get or create a rate limiter for a specific API.
        
        Args:
            api_name: Name of the API (e.g., "openai", "anthropic")
            requests_per_second: Custom requests per second (uses config default if None)
            requests_per_minute: Custom requests per minute (uses config default if None)
            
        Returns:
            TokenBucketRateLimiter instance
        """
        if not self.config.rate_limit_enabled:
            # Return a no-op limiter if rate limiting is disabled
            return _NoOpRateLimiter()
        
        with self.lock:
            if api_name not in self.limiters:
                rps = requests_per_second or self.config.rate_limit_requests_per_second
                self.limiters[api_name] = TokenBucketRateLimiter(
                    requests_per_second=rps,
                    name=api_name
                )
            return self.limiters[api_name]
    
    def get_window_limiter(
        self,
        api_name: str,
        requests_per_minute: Optional[int] = None
    ) -> SlidingWindowRateLimiter:
        """
        Get or create a sliding window rate limiter for a specific API.
        
        Args:
            api_name: Name of the API (e.g., "openai", "anthropic")
            requests_per_minute: Custom requests per minute (uses config default if None)
            
        Returns:
            SlidingWindowRateLimiter instance
        """
        if not self.config.rate_limit_enabled:
            return _NoOpWindowRateLimiter()
        
        with self.lock:
            if api_name not in self.window_limiters:
                rpm = requests_per_minute or self.config.rate_limit_requests_per_minute
                self.window_limiters[api_name] = SlidingWindowRateLimiter(
                    requests_per_minute=rpm,
                    name=api_name
                )
            return self.window_limiters[api_name]
    
    def get_all_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all rate limiters."""
        stats = {}
        with self.lock:
            for name, limiter in self.limiters.items():
                stats[f"{name}_token_bucket"] = limiter.get_stats()
            for name, limiter in self.window_limiters.items():
                stats[f"{name}_window"] = limiter.get_stats()
        return stats


class _NoOpRateLimiter:
    """No-op rate limiter when rate limiting is disabled."""
    
    def acquire(self, tokens: int = 1) -> bool:
        return True
    
    def wait(self, tokens: int = 1) -> float:
        return 0.0
    
    def get_stats(self) -> Dict[str, int]:
        return {"total_requests": 0, "blocked_requests": 0, "current_tokens": 0}


class _NoOpWindowRateLimiter:
    """No-op window rate limiter when rate limiting is disabled."""
    
    def acquire(self) -> bool:
        return True
    
    def wait(self) -> float:
        return 0.0
    
    def get_stats(self) -> Dict[str, int]:
        return {"total_requests": 0, "blocked_requests": 0, "current_window_requests": 0}


# Global rate limiter manager instance
_rate_limiter_manager: Optional[RateLimiterManager] = None
_rate_limiter_lock = threading.Lock()


def get_rate_limiter_manager() -> RateLimiterManager:
    """
    Get or create the global rate limiter manager (thread-safe).
    
    Returns:
        RateLimiterManager instance
    """
    global _rate_limiter_manager
    if _rate_limiter_manager is None:
        with _rate_limiter_lock:
            if _rate_limiter_manager is None:
                _rate_limiter_manager = RateLimiterManager()
    return _rate_limiter_manager

