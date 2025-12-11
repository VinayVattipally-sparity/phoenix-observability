"""
HTTP client with connection pooling for external API calls.

Provides a shared HTTP session with connection pooling to improve
performance and reduce connection overhead.
"""

import logging
import threading
from typing import Optional

from phoenix_observability.config import get_config

logger = logging.getLogger(__name__)

# Try importing requests
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.debug("requests not available. HTTP connection pooling will be disabled.")


class HTTPClientPool:
    """
    HTTP client with connection pooling.
    
    Maintains a shared session with connection pooling for better performance.
    """
    
    def __init__(
        self,
        pool_connections: Optional[int] = None,
        pool_maxsize: Optional[int] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize HTTP client pool.
        
        Args:
            pool_connections: Number of connection pools to cache (defaults to config)
            pool_maxsize: Maximum number of connections to save in the pool (defaults to config)
            timeout: Request timeout in seconds (defaults to config)
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required for HTTP client pooling")
        
        config = get_config()
        self.pool_connections = pool_connections or config.http_pool_connections
        self.pool_maxsize = pool_maxsize or config.http_pool_maxsize
        self.timeout = timeout or config.http_timeout
        
        # Create session with connection pooling
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        
        # Create adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=self.pool_connections,
            pool_maxsize=self.pool_maxsize,
            max_retries=retry_strategy
        )
        
        # Mount adapter for both HTTP and HTTPS
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        logger.info(
            f"HTTP client pool initialized: "
            f"pool_connections={self.pool_connections}, "
            f"pool_maxsize={self.pool_maxsize}, "
            f"timeout={self.timeout}"
        )
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """
        Make a GET request using the pooled session.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments passed to requests.get
            
        Returns:
            Response object
        """
        kwargs.setdefault("timeout", self.timeout)
        return self.session.get(url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """
        Make a POST request using the pooled session.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments passed to requests.post
            
        Returns:
            Response object
        """
        kwargs.setdefault("timeout", self.timeout)
        return self.session.post(url, **kwargs)
    
    def put(self, url: str, **kwargs) -> requests.Response:
        """
        Make a PUT request using the pooled session.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments passed to requests.put
            
        Returns:
            Response object
        """
        kwargs.setdefault("timeout", self.timeout)
        return self.session.put(url, **kwargs)
    
    def delete(self, url: str, **kwargs) -> requests.Response:
        """
        Make a DELETE request using the pooled session.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments passed to requests.delete
            
        Returns:
            Response object
        """
        kwargs.setdefault("timeout", self.timeout)
        return self.session.delete(url, **kwargs)
    
    def close(self):
        """Close the session and release connections."""
        self.session.close()
        logger.debug("HTTP client pool closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global HTTP client pool instance
_http_client_pool: Optional[HTTPClientPool] = None
_http_client_lock = threading.Lock()


def get_http_client() -> HTTPClientPool:
    """
    Get or create the global HTTP client pool (thread-safe).
    
    Returns:
        HTTPClientPool instance
    """
    global _http_client_pool
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library is required for HTTP client pooling")
    
    if _http_client_pool is None:
        with _http_client_lock:
            if _http_client_pool is None:
                _http_client_pool = HTTPClientPool()
    return _http_client_pool


def close_http_client():
    """Close the global HTTP client pool (useful for cleanup)."""
    global _http_client_pool
    with _http_client_lock:
        if _http_client_pool is not None:
            _http_client_pool.close()
            _http_client_pool = None

