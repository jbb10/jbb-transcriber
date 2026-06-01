"""Security validation utilities."""

from __future__ import annotations

from urllib.parse import urlparse

from jbb_transcriber._exceptions import ConfigurationError, SecurityError


def validate_https_url(url: str, *, name: str = "URL") -> None:
    """Validate that a URL uses the HTTPS scheme.

    Args:
        url: The URL to validate.
        name: Human-readable name for error messages (e.g. env var name).

    Raises:
        SecurityError: If the URL does not use HTTPS.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https":
        raise SecurityError(
            f"{name} must use HTTPS (got {parsed.scheme!r}). "
            "Sending API keys over unencrypted connections is not allowed."
        )


def validate_input_size(content: str, max_bytes: int, *, name: str = "input") -> None:
    """Validate that input content does not exceed a size limit.

    Args:
        content: The content to check.
        max_bytes: Maximum allowed size in bytes.
        name: Human-readable name for error messages.

    Raises:
        ConfigurationError: If the content exceeds the limit.
    """
    size = len(content.encode("utf-8"))
    if size > max_bytes:
        raise ConfigurationError([f"{name} is too large ({size:,} bytes, max {max_bytes:,} bytes)"])
