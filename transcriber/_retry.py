"""Shared async retry-with-backoff logic."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

from transcriber._exceptions import LLMError, TranscriptionError

_T = TypeVar("_T")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTTP error classification helpers
# ---------------------------------------------------------------------------

_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def is_transient_http_error(exc: BaseException) -> bool:
    """Determine whether an HTTP-related exception is transient and retryable.

    Classifies errors as follows:

    - ``httpx.TimeoutException`` / ``httpx.ConnectError`` → **retryable**
    - ``httpx.HTTPStatusError`` with status 429/500/502/503/504 → **retryable**
    - ``httpx.HTTPStatusError`` with status 400/401/403/404/… → **not retryable**
    - ``TranscriptionError`` / ``LLMError`` → delegates to ``.is_retryable``
    - Any other ``httpx.HTTPError`` → **retryable** (assume transient)

    Args:
        exc: The exception to classify.

    Returns:
        ``True`` if the error is likely transient and worth retrying.
    """
    if isinstance(exc, (TranscriptionError, LLMError)):
        return exc.is_retryable

    if isinstance(exc, httpx.TimeoutException):
        return True

    if isinstance(exc, httpx.ConnectError):
        return True

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in _TRANSIENT_STATUS_CODES

    # Other httpx.HTTPError subclasses
    if isinstance(exc, httpx.HTTPError):
        return True

    # Unknown exception type — don't retry by default
    return False


def _extract_retry_after(exc: BaseException) -> float | None:
    """Extract ``Retry-After`` header value from an HTTP error, if present.

    Supports integer-seconds values.  HTTP-date values are ignored in favour
    of the standard backoff.

    Args:
        exc: The exception to inspect.

    Returns:
        Delay in seconds, or ``None`` if the header is absent or unparseable.
    """
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    header = getattr(resp, "headers", {}).get("Retry-After")
    if header is None:
        return None
    try:
        return float(header)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Core async retry function
# ---------------------------------------------------------------------------


async def retry_with_backoff(
    fn: Callable[[], Awaitable[_T]],
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    operation_name: str = "operation",
    should_retry: Callable[[BaseException], bool] | None = None,
    jitter: bool = True,
) -> _T:
    """Execute an async callable with exponential backoff on failure.

    Args:
        fn: Zero-argument async callable to execute.
        max_retries: Maximum number of attempts.
        base_delay: Base delay in seconds for the first retry.  Doubles on
            each subsequent attempt (``base_delay``, ``2 * base_delay``, …).
        exceptions: Tuple of exception types to catch and retry on.
        operation_name: Human-readable name for log messages.
        should_retry: Optional predicate that receives the caught exception and
            returns ``True`` if the operation should be retried.  When ``None``
            all caught exceptions trigger a retry.  When the predicate returns
            ``False`` the exception is re-raised immediately (permanent error).
        jitter: Add random jitter (±50 %) to backoff delays to avoid
            thundering-herd effects under concurrent load.

    Returns:
        The return value of ``fn`` on success.

    Raises:
        The last exception raised by ``fn`` after all retries are exhausted,
        or immediately if ``should_retry`` returns ``False``.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                backoff = base_delay * 2 ** (attempt - 1)

                # Honour Retry-After header when available (e.g. 429)
                retry_after = _extract_retry_after(last_exc) if last_exc else None
                if retry_after is not None:
                    backoff = max(backoff, retry_after)
                    logger.info(
                        "Server requested Retry-After: %.1fs for %s",
                        retry_after,
                        operation_name,
                    )

                if jitter:
                    backoff *= random.uniform(0.5, 1.5)  # noqa: S311

                logger.debug(
                    "Retry %s attempt %d/%d after %.1fs backoff",
                    operation_name,
                    attempt + 1,
                    max_retries,
                    backoff,
                )
                await asyncio.sleep(backoff)
            return await fn()
        except exceptions as exc:
            last_exc = exc

            # Log HTTP details when available
            _resp = getattr(exc, "response", None)
            status = getattr(_resp, "status_code", None) if _resp is not None else None
            status_info = f" [HTTP {status}]" if status else ""

            logger.warning(
                "%s failed%s (attempt %d/%d): %s",
                operation_name,
                status_info,
                attempt + 1,
                max_retries,
                exc,
            )

            # Check if the error is permanent (should not be retried)
            if should_retry is not None and not should_retry(exc):
                logger.error(
                    "%s failed with non-retryable error%s — not retrying: %s",
                    operation_name,
                    status_info,
                    exc,
                )
                raise

    # Final failure — log before re-raising
    assert last_exc is not None  # noqa: S101
    logger.error(
        "%s failed after %d attempts — giving up: %s",
        operation_name,
        max_retries,
        last_exc,
    )
    raise last_exc
