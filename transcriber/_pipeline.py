"""Async pipeline — core orchestration for transcription.

This module contains all business logic previously in ``__init__.py``.
Backends are **required** parameters — the pipeline never reads env vars.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from transcriber._audio import get_audio_duration
from transcriber._exceptions import (
    AudioFileError,
    LLMError,
    SynthesisError,
    TranscriberError,
    TranscriptionError,
)
from transcriber._prompts import build_correction_prompt, build_synthesis_prompt
from transcriber._protocols import LLMBackend, TranscriptionBackend
from transcriber._retry import is_transient_http_error, retry_with_backoff
from transcriber._security import validate_input_size
from transcriber._settings import PipelineSettings
from transcriber._types import ChunkResult, CorrectionResult, TranscriptionResult

logger = logging.getLogger(__name__)


async def transcribe(
    path: str | os.PathLike[str],
    *,
    output: str | os.PathLike[str] | None = None,
    glossary: str | os.PathLike[str] | None = None,
    synthesise: bool = False,
    transcription_backend: TranscriptionBackend,
    llm_backend: LLMBackend | None = None,
    settings: PipelineSettings | None = None,
    on_chunk_complete: Callable[[int, int], None] | None = None,
) -> TranscriptionResult:
    """Transcribe an audio or video file (async).

    This is the primary library entry point.

    Args:
        path: Path to the audio/video file.
        output: If given, write the transcript to this file path.
        glossary: Path to a glossary file for LLM-based correction.
        synthesise: Generate a synthesis document alongside the transcript.
        transcription_backend: Backend for audio transcription (required).
        llm_backend: Backend for LLM operations.  Required when ``glossary``
            or ``synthesise`` is specified.
        settings: Pipeline configuration.  Uses defaults if ``None``.
        on_chunk_complete: Optional callback invoked after each chunk finishes.
            Receives ``(chunk_index, total_chunks)``.

    Returns:
        A ``TranscriptionResult`` with the transcript and optional synthesis.

    Raises:
        ConfigurationError: Missing or invalid parameters.
        AudioFileError: File not found or unreadable.
        ConversionError: Format conversion failure.
        TranscriptionError: Transcription API failure.
        LLMError: LLM API failure.
        SynthesisError: Synthesis generation failure.
    """
    if settings is None:
        settings = PipelineSettings()

    audio_path = Path(path)
    if not audio_path.exists():
        raise AudioFileError(f"Audio file not found: {audio_path}", path=str(path))

    # --- Resolve glossary ---
    glossary_text: str | None = None
    if glossary:
        gp = Path(glossary)
        if not gp.exists():
            raise AudioFileError(f"Glossary file not found: {gp}", path=str(glossary))
        glossary_text = gp.read_text(encoding="utf-8")
        validate_input_size(
            glossary_text, settings.max_glossary_size, name="glossary file"
        )

    # --- Transcribe ---
    result_chunks: tuple[ChunkResult, ...] | None = None

    # Audio processing is sync (PyAV) — run in thread
    conv_path, conv_cleanup = await asyncio.to_thread(_open_converted_audio, str(audio_path))
    try:
        duration = await asyncio.to_thread(get_audio_duration, conv_path)

        if duration is None:
            logger.warning(
                "Could not determine audio duration — attempting single-file transcription"
            )

        if duration and duration > settings.max_duration_before_split:
            logger.info("Audio duration: %.1f min — splitting into chunks", duration / 60)

            chunks_list, temp_dir = await asyncio.to_thread(
                _open_split_audio, conv_path, settings.chunk_duration
            )
            try:
                chunk_infos = [
                    (i, chunk, i * settings.chunk_duration)
                    for i, chunk in enumerate(chunks_list)
                ]
                num_workers = min(settings.parallel_workers, len(chunks_list))
                logger.info(
                    "Processing %d chunks with %d parallel workers",
                    len(chunks_list),
                    num_workers,
                )

                chunk_results = await _transcribe_chunks(
                    chunk_infos,
                    transcription_backend,
                    llm_backend,
                    glossary_text,
                    settings,
                    on_chunk_complete,
                )

                chunk_results.sort(key=lambda c: c.index)
                result_chunks = tuple(chunk_results)
                final_transcript = "\n".join(c.transcript for c in chunk_results)
            finally:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            final_transcript = await retry_with_backoff(
                lambda: transcription_backend.transcribe(conv_path),
                max_retries=settings.max_retries,
                base_delay=settings.base_delay,
                exceptions=(TranscriptionError, httpx.HTTPError),
                operation_name="transcription",
                should_retry=is_transient_http_error,
            )

            if glossary_text and llm_backend is not None:
                logger.debug("Applying glossary correction")
                correction = await _correct_with_glossary(
                    final_transcript, glossary_text, llm_backend, settings
                )
                final_transcript = correction.text
    finally:
        if conv_cleanup is not None:
            await asyncio.to_thread(_cleanup_file, conv_cleanup)

    # --- Write output ---
    if output:
        Path(output).write_text(final_transcript, encoding="utf-8")
        logger.debug("Transcript saved to: %s", output)

    # --- Synthesis ---
    synthesis_text: str | None = None
    if synthesise and llm_backend is not None:
        logger.info("Generating synthesis")
        synthesis_text = await synthesise_transcript(
            final_transcript, llm_backend=llm_backend, settings=settings
        )

        if output:
            stem = Path(output).with_suffix("")
            synthesis_path = str(stem) + "_synthesis.md"
            Path(synthesis_path).write_text(synthesis_text, encoding="utf-8")
            logger.debug("Synthesis saved to: %s", synthesis_path)

    return TranscriptionResult(
        transcript=final_transcript,
        synthesis=synthesis_text,
        duration_seconds=duration,
        chunks=result_chunks,
    )


async def synthesise_transcript(
    transcript: str,
    *,
    llm_backend: LLMBackend,
    settings: PipelineSettings | None = None,
) -> str:
    """Generate a synthesis document from a transcript.

    Args:
        transcript: Transcription text.
        llm_backend: An LLM backend instance.
        settings: Pipeline settings for retry configuration.

    Returns:
        Synthesis markdown document.

    Raises:
        SynthesisError: If synthesis fails after all retries.
    """
    if settings is None:
        settings = PipelineSettings()

    prompt = build_synthesis_prompt(transcript)
    try:
        return await retry_with_backoff(
            lambda: llm_backend.complete(prompt, temperature=0.3),
            max_retries=settings.max_retries,
            base_delay=settings.base_delay,
            exceptions=(LLMError, SynthesisError, httpx.HTTPError),
            operation_name="synthesis",
            should_retry=is_transient_http_error,
        )
    except (LLMError, SynthesisError) as e:
        status = getattr(e, "status_code", None)
        body = getattr(e, "response_body", None)
        logger.error(
            "Synthesis generation failed after %d attempts: %s",
            settings.max_retries,
            e,
        )
        raise SynthesisError(
            f"Synthesis failed after {settings.max_retries} attempts: {e}",
            status_code=status,
            response_body=body,
        ) from e


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _transcribe_chunks(
    chunk_infos: list[tuple[int, str, int]],
    backend: TranscriptionBackend,
    llm: LLMBackend | None,
    glossary_text: str | None,
    settings: PipelineSettings,
    on_chunk_complete: Callable[[int, int], None] | None,
) -> list[ChunkResult]:
    """Transcribe multiple chunks with bounded concurrency."""
    semaphore = asyncio.Semaphore(settings.parallel_workers)
    total = len(chunk_infos)

    async def _process(info: tuple[int, str, int]) -> ChunkResult:
        async with semaphore:
            result = await _transcribe_single_chunk(info, backend, llm, glossary_text, settings)
            logger.info("Completed chunk %d/%d", result.index + 1, total)
            if on_chunk_complete is not None:
                on_chunk_complete(result.index, total)
            return result

    return list(await asyncio.gather(*[_process(info) for info in chunk_infos]))


async def _transcribe_single_chunk(
    chunk_info: tuple[int, str, int],
    backend: TranscriptionBackend,
    llm: LLMBackend | None,
    glossary_text: str | None,
    settings: PipelineSettings,
) -> ChunkResult:
    """Transcribe a single chunk and optionally correct it."""
    index, chunk_path, time_offset = chunk_info
    logger.debug("Transcribing chunk %d", index + 1)

    t0 = time.monotonic()
    try:
        text = await retry_with_backoff(
            lambda: backend.transcribe(chunk_path, time_offset=time_offset),
            max_retries=settings.max_retries,
            base_delay=settings.base_delay,
            exceptions=(TranscriptionError, httpx.HTTPError),
            operation_name=f"transcription chunk {index}",
            should_retry=is_transient_http_error,
        )
    except (TranscriptionError, TranscriberError) as exc:
        logger.error("Chunk %d failed permanently: %s", index + 1, exc)
        raise

    if glossary_text and llm is not None:
        logger.debug("Applying glossary correction to chunk %d", index + 1)
        correction = await _correct_with_glossary(text, glossary_text, llm, settings)
        text = correction.text

    elapsed = time.monotonic() - t0
    logger.debug("Chunk %d complete (%.1fs)", index + 1, elapsed)
    return ChunkResult(
        index=index,
        transcript=text,
        start_offset_seconds=float(time_offset),
        processing_time_seconds=elapsed,
    )


async def _correct_with_glossary(
    transcript: str,
    glossary_text: str,
    llm: LLMBackend,
    settings: PipelineSettings,
) -> CorrectionResult:
    """Correct a transcript using an LLM and glossary."""
    prompt = build_correction_prompt(transcript, glossary_text)
    try:
        corrected = await retry_with_backoff(
            lambda: llm.complete(prompt, temperature=0.1),
            max_retries=settings.max_retries,
            base_delay=settings.base_delay,
            exceptions=(LLMError, SynthesisError, httpx.HTTPError),
            operation_name="glossary correction",
            should_retry=is_transient_http_error,
        )
        return CorrectionResult(text=corrected, was_corrected=True)
    except (LLMError, SynthesisError):
        if settings.fail_on_correction_error:
            raise
        logger.warning("Glossary correction failed — using uncorrected text")
        return CorrectionResult(text=transcript, was_corrected=False)


# ---------------------------------------------------------------------------
# Sync audio helpers (called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _open_converted_audio(file_path: str) -> tuple[str, str | None]:
    """Open converted audio and return (conv_path, cleanup_path_or_none).

    Uses the ``converted_audio`` context manager internally, but we need
    to split open/close for the async pipeline.
    """
    from transcriber._audio import API_NO_CONVERSION, _convert_to_m4a

    file_ext = Path(file_path).suffix.lower()
    if file_ext in API_NO_CONVERSION:
        return file_path, None

    converted_path = _convert_to_m4a(file_path)
    return converted_path, converted_path


def _open_split_audio(file_path: str, chunk_duration: int) -> tuple[list[str], str]:
    """Split audio and return (chunks, temp_dir)."""
    from transcriber._audio import _split_audio_file

    return _split_audio_file(file_path, chunk_duration)


def _cleanup_file(path: str) -> None:
    """Remove a file if it exists."""
    if os.path.exists(path):
        os.unlink(path)
