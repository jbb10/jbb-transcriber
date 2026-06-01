"""Transcriber — audio transcription library and CLI tool.

Use as a library::

    import jbb_transcriber

    # Async API (primary)
    result = await jbb_transcriber.transcribe(
        "meeting.mp4",
        transcription_backend=my_backend,
    )

    # Sync convenience wrapper
    result = jbb_transcriber.transcribe_file(
        "meeting.mp4",
        transcription_backend=my_backend,
    )

    # Synthesise an existing transcript
    synthesis = await jbb_transcriber.synthesise_transcript(
        "transcript text...", llm_backend=my_llm,
    )

Use as a CLI::

    transcribe meeting.mp4
    transcribe meeting.mp4 --glossary terms.txt --synthesise
    transcribe --local meeting.mp4
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Coroutine
from typing import Any, TypeVar

from jbb_transcriber._exceptions import (
    AudioFileError,
    ConfigurationError,
    ConversionError,
    LLMError,
    PromptError,
    SecurityError,
    SynthesisError,
    TranscriberError,
    TranscriptionError,
)
from jbb_transcriber._pipeline import synthesise_transcript, transcribe
from jbb_transcriber._prompts import build_correction_prompt, build_synthesis_prompt
from jbb_transcriber._protocols import LLMBackend, TranscriptionBackend
from jbb_transcriber._settings import (
    AzureLLMSettings,
    AzureTranscriptionSettings,
    PipelineSettings,
    WhisperSettings,
)
from jbb_transcriber._types import ChunkResult, CorrectionResult, TranscriptionResult
from jbb_transcriber.backends import (
    AzureLLMBackend,
    AzureTranscriptionBackend,
    WhisperTranscriptionBackend,
    format_whisper_output,
)

# ---------------------------------------------------------------------------
# Version (single source of truth: pyproject.toml)
# ---------------------------------------------------------------------------

try:
    from importlib.metadata import version as _meta_version

    __version__ = _meta_version("jbb-transcriber")
except Exception:  # pragma: no cover — editable installs may not have metadata
    __version__ = "0.0.0-dev"

# ---------------------------------------------------------------------------
# Library logging best practice: add NullHandler so users see nothing unless
# they explicitly configure logging.
# ---------------------------------------------------------------------------

logging.getLogger(__name__).addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Sync convenience wrappers
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an async coroutine synchronously.

    Raises ``RuntimeError`` with a clear message if called from within
    an existing event loop (e.g. Jupyter).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # No loop running — safe to use asyncio.run()
    else:
        raise RuntimeError(
            "transcribe_file() cannot be called from an async context. "
            "Use 'await jbb_transcriber.transcribe(...)' instead."
        )
    return asyncio.run(coro)


def transcribe_file(
    path: str | os.PathLike[str],
    **kwargs: Any,
) -> TranscriptionResult:
    """Sync wrapper around :func:`transcribe`.

    Accepts the same arguments as :func:`transcribe`.
    Cannot be called from within an existing event loop — use the async
    :func:`transcribe` function directly instead.
    """
    return _run_sync(transcribe(path, **kwargs))


def synthesise_text(
    transcript: str,
    **kwargs: Any,
) -> str:
    """Sync wrapper around :func:`synthesise_transcript`."""
    return _run_sync(synthesise_transcript(transcript, **kwargs))


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Async API (primary)
    "transcribe",
    "synthesise_transcript",
    # Sync convenience wrappers
    "transcribe_file",
    "synthesise_text",
    # Result types
    "TranscriptionResult",
    "ChunkResult",
    "CorrectionResult",
    # Settings
    "PipelineSettings",
    "AzureTranscriptionSettings",
    "AzureLLMSettings",
    "WhisperSettings",
    # Exceptions
    "TranscriberError",
    "ConfigurationError",
    "AudioFileError",
    "ConversionError",
    "TranscriptionError",
    "LLMError",
    "SynthesisError",
    "PromptError",
    "SecurityError",
    # Backends
    "AzureTranscriptionBackend",
    "AzureLLMBackend",
    "WhisperTranscriptionBackend",
    # Protocols
    "TranscriptionBackend",
    "LLMBackend",
    # Helpers
    "format_whisper_output",
    "build_correction_prompt",
    "build_synthesis_prompt",
]
