"""
Optional helper to launch Phoenix UI programmatically.

Useful for local development and automated startup scripts.
"""

import logging
import subprocess
import sys
from typing import Optional

from phoenix_observability.config import get_config

logger = logging.getLogger(__name__)


def launch_phoenix(
    port: Optional[int] = None,
    host: str = "localhost",
    background: bool = False,
) -> Optional[subprocess.Popen]:
    """
    Launch Phoenix UI programmatically.

    Args:
        port: Port to run Phoenix on (defaults to 6006)
        host: Host to bind to (defaults to localhost)
        background: If True, run in background process

    Returns:
        Process object if background=True, None otherwise
    """
    if port is None:
        port = 6006

    try:
        # Try to import arize-phoenix
        import phoenix
    except ImportError:
        logger.error(
            "arize-phoenix not installed. Install with: pip install arize-phoenix"
        )
        return None

    cmd = [
        sys.executable,
        "-m",
        "phoenix.server.main",
        "--port",
        str(port),
        "--host",
        host,
    ]

    if background:
        logger.info(f"Starting Phoenix UI in background on {host}:{port}")
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return process
    else:
        logger.info(f"Starting Phoenix UI on {host}:{port}")
        subprocess.run(cmd)
        return None


def stop_phoenix(process: subprocess.Popen) -> None:
    """
    Stop a background Phoenix process.

    Args:
        process: Process object returned from launch_phoenix(background=True)
    """
    if process:
        process.terminate()
        process.wait()
        logger.info("Phoenix UI stopped")

