"""Exception hierarchy for the transcriber package.

All business-logic errors raise typed exceptions instead of calling sys.exit().
Only the CLI layer (cli.py) catches these and converts them to exit codes.
"""

from __future__ import annotations

# Status codes considered transient / retryable
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class TranscriberError(Exception):
    """Base exception for all transcriber errors."""


class ConfigurationError(TranscriberError):
    """Invalid or missing configuration (env vars, parameters, etc.)."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        msg = "; ".join(errors) if len(errors) <= 2 else f"{len(errors)} configuration errors"
        super().__init__(msg)


class AudioFileError(TranscriberError):
    """Audio file not found, unreadable, or has no audio stream."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        self.path = path
        super().__init__(message)


class ConversionError(TranscriberError):
    """Audio/video format conversion failure."""

    def __init__(self, message: str, *, path: str | None = None) -> None:
        self.path = path
        super().__init__(message)


class TranscriptionError(TranscriberError):
    """Transcription API request failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)

    @property
    def is_retryable(self) -> bool:
        """Whether this error is likely transient and worth retrying.

        Returns ``True`` for connection-level failures (no status code) and
        for HTTP status codes 429, 500, 502, 503, 504.  Returns ``False``
        for client errors like 400, 401, 403, 404.
        """
        if self.status_code is None:
            return True  # Connection-level failure — always retry
        return self.status_code in _RETRYABLE_STATUS_CODES


class LLMError(TranscriberError):
    """LLM API request failure (glossary correction or synthesis)."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)

    @property
    def is_retryable(self) -> bool:
        """Whether this error is likely transient and worth retrying."""
        if self.status_code is None:
            return True
        return self.status_code in _RETRYABLE_STATUS_CODES


class SynthesisError(TranscriberError):
    """Synthesis generation failure."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(message)
