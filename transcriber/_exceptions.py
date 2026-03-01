"""Exception hierarchy for the transcriber package.

All business-logic errors raise typed exceptions instead of calling sys.exit().
Only the CLI layer (cli.py) catches these and converts them to exit codes.
"""

from __future__ import annotations


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


class SynthesisError(TranscriberError):
    """Synthesis generation failure."""
