"""Public result types for transcription operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkResult:
    """Metadata for a single processed audio chunk.

    Attributes:
        index: Zero-based chunk index.
        transcript: The transcription text for this chunk.
        start_offset_seconds: Start time of this chunk in the original audio.
        processing_time_seconds: Wall-clock time spent processing this chunk.
    """

    index: int
    transcript: str
    start_offset_seconds: float
    processing_time_seconds: float | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    """Immutable result of a transcription operation.

    Attributes:
        transcript: The transcription text.
        synthesis: The synthesis markdown, or ``None`` if not requested.
        duration_seconds: Audio duration in seconds, or ``None`` if unknown.
        chunks: Per-chunk metadata when the file was split, or ``None``.
    """

    transcript: str
    synthesis: str | None = None
    duration_seconds: float | None = None
    chunks: tuple[ChunkResult, ...] | None = None


@dataclass(frozen=True)
class CorrectionResult:
    """Result of a glossary correction attempt.

    Attributes:
        text: The (possibly corrected) transcript text.
        was_corrected: Whether correction was successfully applied.
    """

    text: str
    was_corrected: bool
