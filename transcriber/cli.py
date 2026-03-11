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
    # Credentials (may be empty in local / synthesise-only mode)
    transcribe_api_key: str
    transcribe_url: str
    text_api_key: str | None
    text_url: str | None


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
    transcribe_api_key = os.getenv("AZURE_TRANSCRIBE_API_KEY")
    transcribe_url = os.getenv("AZURE_TRANSCRIBE_URL")
    text_api_key: str | None = None
    text_url: str | None = None

    if not local and not synthesise_only:
        if not transcribe_api_key:
            errors.append(
                "AZURE_TRANSCRIBE_API_KEY environment variable is not set. "
                'Add to ~/.zshrc: export AZURE_TRANSCRIBE_API_KEY="your-api-key"'
            )
        if not transcribe_url:
            errors.append(
                "AZURE_TRANSCRIBE_URL environment variable is not set. "
                'Add to ~/.zshrc: export AZURE_TRANSCRIBE_URL="your-endpoint-url"'
            )
    else:
        if not transcribe_api_key:
            transcribe_api_key = ""
        if not transcribe_url:
            transcribe_url = ""

    require_text_api = bool(glossary) or synthesise or synthesise_only
    if require_text_api:
        text_api_key = os.getenv("AZURE_TEXT_API_KEY")
        text_url = os.getenv("AZURE_TEXT_URL")

        if glossary:
            feature = "--glossary"
        elif synthesise_only:
            feature = "--synthesise-only"
        else:
            feature = "--synthesise"

        if not text_api_key:
            errors.append(
                f"AZURE_TEXT_API_KEY environment variable is not set (required for {feature}). "
                'Add to ~/.zshrc: export AZURE_TEXT_API_KEY="your-api-key"'
            )
        if not text_url:
            errors.append(
                f"AZURE_TEXT_URL environment variable is not set (required for {feature}). "
                'Add to ~/.zshrc: export AZURE_TEXT_URL="your-endpoint-url"'
            )

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
        transcribe_api_key=transcribe_api_key,  # type: ignore[arg-type]
        transcribe_url=transcribe_url,  # type: ignore[arg-type]
        text_api_key=text_api_key,
        text_url=text_url,
        local_mode=local,
        whisper_model=model,
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
    if validated.text_api_key and validated.text_url:
        llm_settings = AzureLLMSettings(api_key=validated.text_api_key, api_url=validated.text_url)
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
            api_key=validated.transcribe_api_key,
            api_url=validated.transcribe_url,
        )
        transcription_backend = AzureTranscriptionBackend(t_settings)

    try:
        await transcriber.transcribe(
            str(validated.audio_file),
            output=str(validated.output_file),
            glossary=str(validated.glossary) if validated.glossary else None,
            synthesise=validated.synthesise,
            transcription_backend=transcription_backend,
            llm_backend=llm_backend,
            settings=pipeline_settings,
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
  AZURE_TRANSCRIBE_API_KEY    Your Azure OpenAI API key for transcription
  AZURE_TRANSCRIBE_URL        Your Azure OpenAI endpoint URL for transcription
  AZURE_TEXT_API_KEY          Your Azure OpenAI API key for text LLM
                              (required with --glossary, --synthesise, or --synthesise-only)
  AZURE_TEXT_URL              Your Azure OpenAI endpoint URL for text LLM
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
