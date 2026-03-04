"""Shared retry-with-backoff logic."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")

logger = logging.getLogger(__name__)


def retry_with_backoff(
    fn: Callable[[], _T],
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    operation_name: str = "operation",
) -> _T:
    """Execute a callable with exponential backoff on failure.

    Args:
        fn: Zero-argument callable to execute.
        max_retries: Maximum number of attempts.
        base_delay: Base delay in seconds for the first retry.  Doubles on
            each subsequent attempt (``base_delay``, ``2 * base_delay``, …).
        exceptions: Tuple of exception types to catch and retry on.
        operation_name: Human-readable name for log messages.

    Returns:
        The return value of ``fn`` on success.

    Raises:
        The last exception raised by ``fn`` after all retries are exhausted.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                backoff = base_delay * 2 ** (attempt - 1)
                logger.debug(
                    "Retry %s attempt %d/%d after %ds backoff",
                    operation_name,
                    attempt + 1,
                    max_retries,
                    backoff,
                )
                time.sleep(backoff)
            return fn()
        except exceptions as exc:
            last_exc = exc
            logger.warning(
                "%s failed (attempt %d/%d): %s",
                operation_name,
                attempt + 1,
                max_retries,
                exc,
            )

    # Should never be None at this point, but satisfy the type checker
    assert last_exc is not None  # noqa: S101
    raise last_exc
