"""
Async/await utilities for I/O operations.

Provides async wrappers for external API calls and utilities
for converting between sync and async code.
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Callable, Coroutine, TypeVar, Optional

logger = logging.getLogger(__name__)

T = TypeVar('T')


def async_to_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Convert an async coroutine to sync execution.
    
    This is useful when you need to call async code from sync context.
    Creates a new event loop if one doesn't exist.
    
    Args:
        coro: Coroutine to execute
        
    Returns:
        Result of the coroutine
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, we can't use it
            # Create a new thread with a new event loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)


def sync_to_async(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Convert a sync function to async.
    
    This wraps a sync function to run in an executor, making it awaitable.
    
    Args:
        func: Sync function to wrap
        
    Returns:
        Async function that wraps the sync function
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    
    return wrapper


async def gather_with_concurrency(
    limit: int,
    *coros: Coroutine[Any, Any, Any]
) -> list[Any]:
    """
    Run coroutines with a concurrency limit.
    
    This is useful for rate limiting async operations.
    
    Args:
        limit: Maximum number of concurrent coroutines
        *coros: Coroutines to run
        
    Returns:
        List of results in the same order as coros
    """
    semaphore = asyncio.Semaphore(limit)
    
    async def bounded_coro(coro: Coroutine[Any, Any, Any]) -> Any:
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*(bounded_coro(coro) for coro in coros))


class AsyncRateLimiter:
    """
    Async rate limiter using asyncio.
    
    Provides token bucket rate limiting for async operations.
    """
    
    def __init__(self, requests_per_second: int, burst_size: Optional[int] = None):
        """
        Initialize async rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst size (defaults to requests_per_second / 2)
        """
        self.requests_per_second = max(1, requests_per_second)
        self.burst_size = burst_size or max(1, self.requests_per_second // 2)
        self.tokens = float(self.burst_size)
        self.last_refill = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens (non-blocking).
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
            
        Returns:
            True if request is allowed, False if rate limited
        """
        async with self.lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_refill
            
            # Refill tokens based on elapsed time
            refill_amount = elapsed * self.requests_per_second
            self.tokens = min(self.burst_size, self.tokens + refill_amount)
            self.last_refill = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            else:
                return False
    
    async def wait(self, tokens: int = 1) -> float:
        """
        Wait until tokens are available, then acquire them.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
            
        Returns:
            Wait time in seconds
        """
        wait_time = 0.0
        while not await self.acquire(tokens):
            # Calculate how long to wait
            async with self.lock:
                needed = tokens - self.tokens
                wait_time = needed / self.requests_per_second
                if wait_time < 0.01:
                    wait_time = 0.01
            
            await asyncio.sleep(wait_time)
            wait_time += wait_time
        
        return wait_time

