"""Command-line interface for the transcriber.

This module is the **only** place that calls ``sys.exit()``, touches
``argparse``, or configures logging output formatting.  All business
logic is accessed through the public ``transcriber`` package API.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import transcriber
from transcriber._settings import (
    AzureLLMSettings,
    AzureTranscriptionSettings,
    PipelineSettings,
    WhisperSettings,
)
from transcriber.backends import (
    AzureLLMBackend,
    AzureTranscriptionBackend,
    WhisperTranscriptionBackend,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging configuration (CLI only)
# ---------------------------------------------------------------------------


class _CLIFormatter(logging.Formatter):
    """CLI formatter with optional ANSI colour for warnings and errors.

    Output style::

        [2026-03-04 13:20:04] Splitting audio into 15-minute chunks
        [2026-03-04 13:20:06] WARNING: Glossary correction failed — using uncorrected text
        [2026-03-04 13:20:06] ERROR: API request failed: 500
    """

    _RESET = "\033[0m"
    _LEVEL_COLOURS = {
        logging.WARNING: "\033[33m",  # yellow
        logging.ERROR: "\033[31m",  # red
        logging.CRITICAL: "\033[31;1m",  # bold red
    }

    def __init__(self, *, use_colour: bool = False) -> None:
        super().__init__()
        self._use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = record.getMessage()

        if record.levelno >= logging.WARNING:
            label = record.levelname
            colour = self._LEVEL_COLOURS.get(record.levelno, "")
            if self._use_colour and colour:
                return f"[{timestamp}] {colour}{label}: {msg}{self._RESET}"
            return f"[{timestamp}] {label}: {msg}"

        return f"[{timestamp}] {msg}"


def _setup_cli_logging() -> None:
    """Configure logging to emit to stderr with the CLI timestamp format."""
    root = logging.getLogger("transcriber")
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in root.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        use_colour = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
        handler.setFormatter(_CLIFormatter(use_colour=use_colour))
        root.addHandler(handler)
        root.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# CLI-specific validated configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedConfig:
    """Validated configuration for a CLI transcription job.

    This is a CLI concern — library consumers use ``transcribe()``
    and ``synthesise_transcript()`` directly and never see this type.
    """

    audio_file: Path
    output_file: Path
    glossary: Path | None
    glossary_text: str | None
    synthesise: bool
    synthesise_only: bool
    parallel_workers: int
    chunk_duration: int
    # Provider selection
    provider: str
    # Local mode
    local_mode: bool
    whisper_model: str
    # Shared LiteLLM proxy credentials (empty in local mode)
    api_key: str
    base_url: str
    transcribe_model: str
    text_model: str | None


def validate_cli_config(  # noqa: C901 — complexity is inherent to validation
    *,
    audio_file: str,
    output_file: str | None,
    glossary: str | None,
    synthesise: bool,
    synthesise_only: bool,
    parallel_workers: int,
    chunk_duration: int = 900,
    local: bool,
    model: str,
    provider: str = "azure",
) -> ValidatedConfig:
    """Validate CLI parameters and environment variables.

    Collects **all** validation errors before raising, so users can fix
    everything at once.

    Args:
        audio_file: Positional CLI argument (audio or transcript path).
        output_file: Optional output path.
        glossary: Optional glossary file path.
        synthesise: Whether ``--synthesise`` was passed.
        synthesise_only: Whether ``--synthesise-only`` was passed.
        parallel_workers: Max parallel workers.
        chunk_duration: Seconds per audio chunk.
        local: Whether ``--local`` was passed.
        model: Whisper model name.
        provider: Backend provider name (default ``"azure"``).

    Returns:
        A fully validated ``ValidatedConfig``.

    Raises:
        ConfigurationError: If any validation fails.
    """
    from transcriber._audio import is_text_file, log_audio_file_info, probe_audio_stream

    errors: list[str] = []

    # --- Conflicting flags ---
    if synthesise_only and synthesise:
        errors.append("Cannot use --synthesise and --synthesise-only together")

    # --- Paths ---
    audio_path = Path(audio_file)

    # --- Auto-detect text files → synthesis-only mode ---
    if not synthesise_only and is_text_file(audio_path):
        synthesise_only = True
        synthesise = False  # clear to avoid the conflict check
        logger.info("Text file detected — skipping transcription and running synthesis only.")

    if output_file is None:
        out_path = audio_path.with_suffix(".txt")
    else:
        out_path = Path(output_file)

    # --- parallel_workers ---
    if parallel_workers > 100:
        errors.append(f"--parallel-workers cannot exceed 100, got {parallel_workers}")

    # --- chunk_duration ---
    if chunk_duration < 60:
        errors.append(f"--chunk-duration must be at least 60 seconds, got {chunk_duration}")
    elif chunk_duration > 3600:
        errors.append(f"--chunk-duration cannot exceed 3600 seconds, got {chunk_duration}")

    # --- File validation ---
    if synthesise_only:
        if not audio_path.exists():
            errors.append(f"Transcript file not found: {audio_path}")
        elif not audio_path.is_file():
            errors.append(f"Transcript path is not a file: {audio_path}")
    else:
        if not audio_path.exists():
            errors.append(f"Audio file not found: {audio_path}")
        elif not audio_path.is_file():
            errors.append(f"Audio path is not a file: {audio_path}")
        else:
            has_audio, audio_error = probe_audio_stream(audio_path)
            if not has_audio:
                errors.append(audio_error or f"No audio stream in: {audio_path}")

    # --- Output directory ---
    output_dir = out_path.parent
    if output_dir and str(output_dir) != ".":
        if not output_dir.exists():
            errors.append(f"Output directory does not exist: {output_dir}")
        elif not os.access(output_dir, os.W_OK):
            errors.append(f"Output directory is not writable: {output_dir}")

    # --- Glossary ---
    glossary_path: Path | None = None
    glossary_text: str | None = None
    if glossary:
        glossary_path = Path(glossary)
        if not glossary_path.exists():
            errors.append(f"Glossary file not found: {glossary_path}")
        elif not glossary_path.is_file():
            errors.append(f"Glossary path is not a file: {glossary_path}")

    # --- Environment variables ---
    # Credentials: TRANSCRIBER_* overrides OPENAI_* (per-app spend tracking)
    from transcriber._settings import resolve_api_key, resolve_base_url

    api_key: str = ""
    base_url: str = ""
    transcribe_model: str = ""
    text_model: str | None = None

    require_text_features = bool(glossary) or synthesise or synthesise_only

    # API credentials are needed when:
    #   - running cloud transcription (not local), OR
    #   - using any text feature (glossary/synthesise) even in local mode
    need_api_credentials = (not local) or require_text_features
    if need_api_credentials:
        _api_key = resolve_api_key()
        _base_url = resolve_base_url()

        if not _api_key:
            errors.append(
                "Neither TRANSCRIBER_API_KEY nor OPENAI_API_KEY is set. "
                'Add to ~/.zshrc: export TRANSCRIBER_API_KEY="your-litellm-key"'
            )
        else:
            api_key = _api_key

        if not _base_url:
            errors.append(
                "Neither TRANSCRIBER_BASE_URL nor OPENAI_BASE_URL is set. "
                'Add to ~/.zshrc: export TRANSCRIBER_BASE_URL="https://your-proxy.example.com/v1"'
            )
        else:
            base_url = _base_url

    if not local and not synthesise_only:
        _transcribe_model = os.getenv("TRANSCRIBER_MODEL")
        if not _transcribe_model:
            errors.append(
                "TRANSCRIBER_MODEL environment variable is not set. "
                'Add to ~/.zshrc: export TRANSCRIBER_MODEL="your-transcription-model-name"'
            )
        else:
            transcribe_model = _transcribe_model
    elif not local:
        transcribe_model = os.getenv("TRANSCRIBER_MODEL", "")

    if require_text_features:
        if glossary:
            feature = "--glossary"
        elif synthesise_only:
            feature = "--synthesise-only"
        else:
            feature = "--synthesise"

        _text_model = os.getenv("TRANSCRIBER_TEXT_MODEL")
        if not _text_model:
            errors.append(
                f"TRANSCRIBER_TEXT_MODEL environment variable is not set (required for {feature}). "
                'Add to ~/.zshrc: export TRANSCRIBER_TEXT_MODEL="your-text-model-name"'
            )
        else:
            text_model = _text_model

    # --- Whisper availability ---
    if local:
        try:
            import whisper  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            errors.append(
                "openai-whisper is not installed (required for --local mode). "
                'Install with: uv tool install "transcriber[local]"'
            )

    # --- Raise all errors at once ---
    if errors:
        raise transcriber.ConfigurationError(errors)

    # --- Load glossary text (validation passed) ---
    if glossary_path:
        try:
            glossary_text = glossary_path.read_text(encoding="utf-8")
            logger.debug("Loaded glossary from: %s", glossary_path)
        except OSError as e:
            raise transcriber.ConfigurationError([f"Could not read glossary file: {e}"]) from e

    # Log audio file info
    if not synthesise_only:
        log_audio_file_info(audio_path)

    return ValidatedConfig(
        audio_file=audio_path,
        output_file=out_path,
        glossary=glossary_path,
        glossary_text=glossary_text,
        synthesise=synthesise,
        synthesise_only=synthesise_only,
        parallel_workers=parallel_workers,
        chunk_duration=chunk_duration,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        local_mode=local,
        whisper_model=model,
        transcribe_model=transcribe_model,
        text_model=text_model,
    )


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------


def positive_int(value: str) -> int:
    """Argparse type for positive integers (>= 1)."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: '{value}'") from None
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return ivalue


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _write_output(text: str, output_file_path: str) -> None:
    """Write text to a file.

    Raises:
        TranscriberError: If the file cannot be written.
    """
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Saved to: %s", output_file_path)
    except OSError as e:
        raise transcriber.TranscriberError(f"Could not write to output file: {e}") from e


# ---------------------------------------------------------------------------
# Main orchestration — delegates to the library's public API
# ---------------------------------------------------------------------------


def _run(validated: ValidatedConfig) -> None:
    """Execute the transcription pipeline with a validated config.

    Builds backend instances from ``validated`` and delegates to the
    library-level async pipeline.
    """
    asyncio.run(_run_async(validated))


async def _run_async(validated: ValidatedConfig) -> None:
    """Async implementation of the CLI pipeline."""
    pipeline_settings = PipelineSettings(
        chunk_duration=validated.chunk_duration,
        parallel_workers=validated.parallel_workers,
    )

    # Build LLM backend from validated credentials
    llm_backend: AzureLLMBackend | None = None
    if validated.text_model and validated.api_key and validated.base_url:
        llm_settings = AzureLLMSettings(
            api_key=validated.api_key,
            api_url=validated.base_url,
            model=validated.text_model,
        )
        llm_backend = AzureLLMBackend(llm_settings)

    # --- synthesise-only mode ---
    if validated.synthesise_only:
        logger.info("Reading existing transcript from: %s", validated.audio_file)
        transcript_text = validated.audio_file.read_text(encoding="utf-8")

        if not transcript_text.strip():
            raise transcriber.ConfigurationError(["Transcript file is empty"])

        if llm_backend is None:
            raise transcriber.ConfigurationError(["LLM backend required for synthesis"])

        logger.info("Generating synthesis document...")
        try:
            synthesis = await transcriber.synthesise_transcript(
                transcript_text, llm_backend=llm_backend, settings=pipeline_settings
            )
            output_stem = validated.audio_file.with_suffix("")
            _write_output(synthesis, str(output_stem) + "_synthesis.md")
            logger.info("Synthesis complete!")
        finally:
            await llm_backend.aclose()
        return

    # --- Normal transcription ---
    transcription_backend: transcriber.TranscriptionBackend
    if validated.local_mode:
        whisper_settings = WhisperSettings(model_name=validated.whisper_model)
        transcription_backend = WhisperTranscriptionBackend(settings=whisper_settings)
    else:
        t_settings = AzureTranscriptionSettings(
            api_key=validated.api_key,
            api_url=validated.base_url,
            model=validated.transcribe_model,
        )
        transcription_backend = AzureTranscriptionBackend(t_settings)

    try:
        completed: list[int] = []

        def _on_chunk_done(chunk_idx: int, total: int) -> None:
            completed.append(chunk_idx)
            logger.info("Chunk %d/%d complete", len(completed), total)

        await transcriber.transcribe(
            str(validated.audio_file),
            output=str(validated.output_file),
            glossary=str(validated.glossary) if validated.glossary else None,
            synthesise=validated.synthesise,
            transcription_backend=transcription_backend,
            llm_backend=llm_backend,
            settings=pipeline_settings,
            on_chunk_complete=_on_chunk_done,
        )
    except transcriber.SynthesisError as e:
        logger.warning("%s", e)
        logger.warning("Transcript was saved successfully, but synthesis could not be generated.")
    finally:
        if isinstance(transcription_backend, AzureTranscriptionBackend):
            await transcription_backend.aclose()
        if llm_backend is not None:
            await llm_backend.aclose()

    logger.info("Transcription complete!")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main CLI entry point."""
    _setup_cli_logging()

    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Azure OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  TRANSCRIBER_API_KEY         LiteLLM virtual key (overrides OPENAI_API_KEY)
  OPENAI_API_KEY              Fallback API key if TRANSCRIBER_API_KEY is not set
  TRANSCRIBER_BASE_URL        LiteLLM proxy base URL (overrides OPENAI_BASE_URL)
  OPENAI_BASE_URL             Fallback base URL if TRANSCRIBER_BASE_URL is not set
  TRANSCRIBER_MODEL           Model name for transcription (e.g., gpt-4o-transcribe)
  TRANSCRIBER_TEXT_MODEL      Model name for text LLM
                              (required with --glossary, --synthesise, or --synthesise-only)

Example:
  transcribe audio.mp3 transcript.txt
  transcribe podcast.m4a output.txt
  transcribe video.mp4
  (output defaults to video.txt if not specified)

With glossary correction:
  transcribe audio.mp3 transcript.txt --glossary terms.txt
  transcribe long_recording.m4a output.txt --glossary company_terms.txt --parallel-workers 10

With synthesis:
  transcribe meeting.mp4 --synthesise
  transcribe meeting.mp4 --glossary terms.txt --synthesise
  (creates both meeting.txt and meeting_synthesis.md)

Synthesise an existing transcript:
  transcribe meeting.txt --synthesise-only
  transcribe meeting.txt
  (text files are auto-detected — synthesis runs without needing --synthesise-only)
  (reads meeting.txt and creates meeting_synthesis.md)

Note: Files longer than 25 minutes will be automatically split into chunks.
        """,
    )

    parser.add_argument("audio_file", help="Path to the audio file to transcribe")

    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help="Path to output file (optional - defaults to input with .txt extension)",
    )

    parser.add_argument(
        "--glossary",
        "-g",
        help="Path to glossary file with terms/names/acronyms for transcript correction",
    )

    parser.add_argument(
        "--synthesise",
        "-s",
        action="store_true",
        help="Generate a synthesis document (markdown) summarising the transcript",
    )

    parser.add_argument(
        "--synthesise-only",
        "-S",
        action="store_true",
        help="Generate a synthesis from an existing transcript file (skips transcription)",
    )

    parser.add_argument(
        "--parallel-workers",
        "-p",
        type=positive_int,
        default=15,
        help="Maximum number of parallel workers for processing chunks (default: 15, max: 100)",
    )

    parser.add_argument(
        "--chunk-duration",
        type=positive_int,
        default=900,
        help="Duration in seconds for each audio chunk when splitting long files "
        "(default: 900 = 15 min, min: 60, max: 3600)",
    )

    parser.add_argument(
        "--local",
        "-l",
        action="store_true",
        help="Use local Whisper model instead of Azure OpenAI "
        "(requires openai-whisper: uv tool install 'transcriber[local]')",
    )

    parser.add_argument(
        "--model",
        default="base",
        choices=[
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v1",
            "large-v2",
            "large-v3",
            "turbo",
        ],
        help="Whisper model to use with local mode (default: base). "
        "Valid models: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, "
        "large, large-v1, large-v2, large-v3, turbo",
    )

    parser.add_argument(
        "--provider",
        default="azure",
        choices=["azure"],
        help="Backend provider for cloud transcription (default: azure)",
    )

    args = parser.parse_args()

    try:
        validated = validate_cli_config(
            audio_file=args.audio_file,
            output_file=args.output_file,
            glossary=args.glossary,
            synthesise=args.synthesise,
            synthesise_only=getattr(args, "synthesise_only", False),
            parallel_workers=args.parallel_workers,
            chunk_duration=args.chunk_duration,
            local=args.local,
            model=args.model,
            provider=args.provider,
        )
        _run(validated)
    except transcriber.ConfigurationError as e:
        for err in e.errors:
            logger.error("Error: %s", err)
        sys.exit(1)
    except transcriber.TranscriberError as e:
        logger.error("Error: %s", e)
        sys.exit(1)
