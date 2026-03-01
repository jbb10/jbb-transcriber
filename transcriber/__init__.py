"""Transcriber — audio transcription library and CLI tool.

Use as a library::

    import transcriber

    # Cloud transcription (reads API credentials from env vars)
    result = transcriber.transcribe_file("meeting.mp4")

    # With explicit backend (no env vars needed)
    backend = transcriber.AzureTranscriptionBackend(api_key="...", api_url="...")
    result = transcriber.transcribe_file("meeting.mp4", transcription_backend=backend)

    # Local Whisper (requires ``pip install transcriber[local]``)
    result = transcriber.transcribe_file("meeting.mp4", local=True, model="medium")

    # Synthesise an existing transcript
    synthesis = transcriber.synthesise_text("transcript text...")

Use as a CLI::

    transcribe meeting.mp4
    transcribe meeting.mp4 --glossary terms.txt --synthesise
    transcribe --local meeting.mp4
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from transcriber._audio import converted_audio, get_audio_duration, split_audio
from transcriber._exceptions import (
    AudioFileError,
    ConfigurationError,
    ConversionError,
    SynthesisError,
    TranscriberError,
    TranscriptionError,
)
from transcriber._llm import (
    AzureLLMBackend,
    build_correction_prompt,
    build_synthesis_prompt,
    correct_with_glossary,
    synthesise_transcript,
)
from transcriber._protocols import LLMBackend, TranscriptionBackend
from transcriber._transcription import (
    AzureTranscriptionBackend,
    WhisperTranscriptionBackend,
    format_whisper_output,
)

# ---------------------------------------------------------------------------
# Version (single source of truth: pyproject.toml)
# ---------------------------------------------------------------------------

try:
    from importlib.metadata import version as _meta_version

    __version__ = _meta_version("transcriber")
except Exception:  # pragma: no cover — editable installs may not have metadata
    __version__ = "0.0.0-dev"

# ---------------------------------------------------------------------------
# Library logging best practice: add NullHandler so users see nothing unless
# they explicitly configure logging.
# ---------------------------------------------------------------------------

logging.getLogger(__name__).addHandler(logging.NullHandler())

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TranscriptionResult:
    """Immutable result of a transcription operation.

    Attributes:
        transcript: The transcription text.
        synthesis: The synthesis markdown, or ``None`` if not requested.
        duration_seconds: Audio duration in seconds, or ``None`` if unknown.
    """

    transcript: str
    synthesis: str | None = None
    duration_seconds: float | None = None


# ---------------------------------------------------------------------------
# High-level public API
# ---------------------------------------------------------------------------


def transcribe_file(  # noqa: C901
    path: str,
    *,
    output: str | None = None,
    glossary: str | None = None,
    synthesise: bool = False,
    local: bool = False,
    model: str = "base",
    parallel_workers: int = 15,
    transcription_backend: TranscriptionBackend | None = None,
    llm_backend: LLMBackend | None = None,
) -> TranscriptionResult:
    """Transcribe an audio or video file.

    This is the main library entry point.  Call it from your own Python code
    to transcribe audio without going through the CLI.

    Args:
        path: Path to the audio/video file.
        output: If given, write the transcript to this file path.
        glossary: Path to a glossary file for LLM-based correction.
        synthesise: Generate a synthesis document alongside the transcript.
        local: Use local Whisper model instead of Azure API.
        model: Whisper model name (only used when ``local=True``).
        parallel_workers: Max parallel workers for long-file chunking.
        transcription_backend: Custom transcription backend.  If ``None``,
            an ``AzureTranscriptionBackend`` (or ``WhisperTranscriptionBackend``
            when ``local=True``) is created automatically.
        llm_backend: Custom LLM backend.  If ``None`` and an LLM is needed
            (glossary or synthesis), an ``AzureLLMBackend`` is created from
            environment variables.

    Returns:
        A ``TranscriptionResult`` with the transcript and optional synthesis.

    Raises:
        ConfigurationError: Missing credentials or invalid parameters.
        AudioFileError: File not found or unreadable.
        ConversionError: Format conversion failure.
        TranscriptionError: Transcription API failure.
        SynthesisError: Synthesis generation failure.
    """
    from pathlib import Path as _Path

    audio_path = _Path(path)
    if not audio_path.exists():
        raise AudioFileError(f"Audio file not found: {audio_path}", path=path)

    # --- Resolve glossary ---
    glossary_text: str | None = None
    if glossary:
        gp = _Path(glossary)
        if not gp.exists():
            raise AudioFileError(f"Glossary file not found: {gp}", path=glossary)
        glossary_text = gp.read_text(encoding="utf-8")

    # --- Resolve LLM backend ---
    need_llm = glossary_text is not None or synthesise
    resolved_llm: LLMBackend | None = llm_backend
    if need_llm and resolved_llm is None:
        resolved_llm = AzureLLMBackend.from_env()

    # --- Resolve transcription backend ---
    resolved_transcription: TranscriptionBackend
    if transcription_backend is not None:
        resolved_transcription = transcription_backend
    elif local:
        resolved_transcription = WhisperTranscriptionBackend(model_name=model)
    else:
        resolved_transcription = AzureTranscriptionBackend.from_env()

    # --- Transcribe ---
    with converted_audio(str(audio_path)) as conv_path:
        duration = get_audio_duration(conv_path)
        max_duration = 1400

        if (
            not local
            and duration
            and duration > max_duration
            and isinstance(resolved_transcription, AzureTranscriptionBackend)
        ):
            logger.info("Audio duration: %.1f min (exceeds limit)", duration / 60)

            with split_audio(conv_path, chunk_duration=900) as chunks:
                chunk_dur = 900
                chunk_infos = [(i, chunk, i * chunk_dur) for i, chunk in enumerate(chunks)]
                num_workers = min(parallel_workers, len(chunks))
                logger.info(
                    "Processing %d chunks with %d parallel workers...",
                    len(chunks),
                    num_workers,
                )

                results: dict[int, str] = {}
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    future_to_idx = {
                        executor.submit(
                            _transcribe_chunk,
                            info,
                            resolved_transcription,
                            resolved_llm,
                            glossary_text,
                        ): info[0]
                        for info in chunk_infos
                    }
                    for future in as_completed(future_to_idx):
                        idx, text = future.result()
                        results[idx] = text
                        logger.info("Completed chunk %d/%d", idx + 1, len(chunks))

                all_transcriptions = [results[i] for i in range(len(chunks))]
                final_transcript = "\n".join(all_transcriptions)
        else:
            final_transcript = resolved_transcription.transcribe(conv_path)

            if glossary_text and resolved_llm is not None:
                logger.info("Applying glossary correction...")
                final_transcript = correct_with_glossary(
                    final_transcript, glossary_text, resolved_llm
                )

    # --- Write output ---
    if output:
        _Path(output).write_text(final_transcript, encoding="utf-8")
        logger.info("Transcript saved to: %s", output)

    # --- Synthesis ---
    synthesis_text: str | None = None
    if synthesise and resolved_llm is not None:
        logger.info("Generating synthesis document...")
        synthesis_text = synthesise_transcript(final_transcript, resolved_llm)

        if output:
            stem = _Path(output).with_suffix("")
            synthesis_path = str(stem) + "_synthesis.md"
            _Path(synthesis_path).write_text(synthesis_text, encoding="utf-8")
            logger.info("Synthesis saved to: %s", synthesis_path)

    return TranscriptionResult(
        transcript=final_transcript,
        synthesis=synthesis_text,
        duration_seconds=duration,
    )


def synthesise_text(
    transcript: str,
    *,
    llm_backend: LLMBackend | None = None,
) -> str:
    """Generate a synthesis document from transcript text.

    Args:
        transcript: The transcript text to synthesise.
        llm_backend: Custom LLM backend.  If ``None``, an
            ``AzureLLMBackend`` is created from environment variables.

    Returns:
        Synthesis markdown text.

    Raises:
        ConfigurationError: If LLM credentials are missing.
        SynthesisError: If synthesis fails.
    """
    resolved_llm: LLMBackend = llm_backend or AzureLLMBackend.from_env()
    return synthesise_transcript(transcript, resolved_llm)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _transcribe_chunk(
    chunk_info: tuple[int, str, int],
    backend: TranscriptionBackend,
    llm: LLMBackend | None,
    glossary_text: str | None,
) -> tuple[int, str]:
    """Transcribe a single chunk and optionally correct it."""
    index, chunk_path, time_offset = chunk_info
    logger.info("Transcribing chunk %d...", index + 1)

    text = backend.transcribe(chunk_path, time_offset=time_offset)

    if glossary_text and llm is not None:
        logger.info("Applying glossary correction to chunk %d...", index + 1)
        text = correct_with_glossary(text, glossary_text, llm)

    logger.info("Chunk %d complete", index + 1)
    return (index, text)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # High-level functions
    "transcribe_file",
    "synthesise_text",
    # Result type
    "TranscriptionResult",
    # Exceptions
    "TranscriberError",
    "ConfigurationError",
    "AudioFileError",
    "ConversionError",
    "TranscriptionError",
    "SynthesisError",
    # Backends
    "AzureTranscriptionBackend",
    "AzureLLMBackend",
    "WhisperTranscriptionBackend",
    # Protocols
    "TranscriptionBackend",
    "LLMBackend",
    # Helpers (useful for library users)
    "format_whisper_output",
    "build_correction_prompt",
    "build_synthesis_prompt",
]
