"""Command-line interface for the transcriber.

This is the **only** module that calls ``sys.exit()``.  All business logic
raises typed exceptions from ``transcriber._exceptions``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from transcriber._audio import converted_audio, get_audio_duration, split_audio
from transcriber._config import ValidatedConfig, validate_cli_config
from transcriber._exceptions import ConfigurationError, SynthesisError, TranscriberError
from transcriber._llm import AzureLLMBackend, correct_with_glossary, synthesise_transcript
from transcriber._transcription import AzureTranscriptionBackend, WhisperTranscriptionBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging configuration (CLI only)
# ---------------------------------------------------------------------------


class _CLIFormatter(logging.Formatter):
    """Formatter that reproduces the original ``[YYYY-MM-DD HH:MM:SS] msg`` style."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] {record.getMessage()}"


def _setup_cli_logging() -> None:
    """Configure logging to emit to stderr with the CLI timestamp format."""
    root = logging.getLogger("transcriber")
    # Remove the NullHandler added by the library __init__ and install a
    # user-facing StreamHandler so that CLI messages appear on stderr.
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in root.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_CLIFormatter())
        root.addHandler(handler)
        root.setLevel(logging.INFO)


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
    """Write text to a file."""
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info("Transcription saved to: %s", output_file_path)
    except OSError as e:
        logger.error("Could not write to output file: %s", e)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Chunk processing helper
# ---------------------------------------------------------------------------


def _process_chunk(
    chunk_info: tuple[int, str, int],
    transcription_backend: AzureTranscriptionBackend,
    llm_backend: AzureLLMBackend | None,
    glossary_text: str | None,
) -> tuple[int, str]:
    """Process a single audio chunk: transcribe and optionally correct."""
    index, chunk_path, time_offset = chunk_info

    logger.info("Transcribing chunk %d...", index + 1)
    transcription = transcription_backend.transcribe(chunk_path, time_offset=time_offset)

    if glossary_text and llm_backend is not None:
        logger.info("Applying glossary correction to chunk %d...", index + 1)
        transcription = correct_with_glossary(transcription, glossary_text, llm_backend)

    logger.info("Chunk %d complete", index + 1)
    return (index, transcription)


# ---------------------------------------------------------------------------
# Main orchestration (extracted from old main() for testability)
# ---------------------------------------------------------------------------


def _run(validated: ValidatedConfig) -> None:  # noqa: C901
    """Execute the transcription pipeline with a validated config.

    This function contains the core orchestration logic and raises exceptions
    instead of calling sys.exit().
    """
    # Build backends from validated config
    llm_backend: AzureLLMBackend | None = None
    if validated.text_api_key and validated.text_url:
        llm_backend = AzureLLMBackend(validated.text_api_key, validated.text_url)

    # --- synthesise-only mode ---
    if validated.synthesise_only:
        transcript_path = validated.audio_file
        logger.info("Reading existing transcript from: %s", transcript_path)
        transcript_text = transcript_path.read_text(encoding="utf-8")

        if not transcript_text.strip():
            logger.error("Transcript file is empty")
            sys.exit(1)

        output_stem = transcript_path.with_suffix("")
        synthesis_output = str(output_stem) + "_synthesis.md"
        logger.info("Generating synthesis document...")

        if llm_backend is None:
            logger.error("LLM backend not configured")
            sys.exit(1)

        synthesis = synthesise_transcript(transcript_text, llm_backend)
        _write_output(synthesis, synthesis_output)
        logger.info("Synthesis complete!")
        return

    # --- Normal transcription ---
    with converted_audio(str(validated.audio_file)) as audio_path:
        if validated.local_mode:
            logger.info("Using local Whisper model for transcription")
            whisper_backend = WhisperTranscriptionBackend(
                model_name=validated.whisper_model,
            )
            final_transcript = whisper_backend.transcribe(audio_path)

            if validated.glossary_text and llm_backend is not None:
                logger.info("Applying glossary correction...")
                final_transcript = correct_with_glossary(
                    final_transcript, validated.glossary_text, llm_backend
                )

            _write_output(final_transcript, str(validated.output_file))
        else:
            # Azure API transcription
            transcription_backend = AzureTranscriptionBackend(
                validated.transcribe_api_key, validated.transcribe_url
            )

            duration = get_audio_duration(audio_path)
            max_duration = 1400  # ~23 min 20 s

            if duration and duration > max_duration:
                logger.info("Audio duration: %.1f min (exceeds limit)", duration / 60)

                with split_audio(audio_path, chunk_duration=900) as chunks:
                    chunk_duration = 900
                    chunk_infos = [(i, chunk, i * chunk_duration) for i, chunk in enumerate(chunks)]

                    num_workers = min(validated.parallel_workers, len(chunks))
                    logger.info(
                        "Processing %d chunks with %d parallel workers...",
                        len(chunks),
                        num_workers,
                    )

                    results: dict[int, str] = {}
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        future_to_index = {
                            executor.submit(
                                _process_chunk,
                                info,
                                transcription_backend,
                                llm_backend,
                                validated.glossary_text,
                            ): info[0]
                            for info in chunk_infos
                        }

                        for future in as_completed(future_to_index):
                            index = future_to_index[future]
                            try:
                                idx, transcription = future.result()
                                results[idx] = transcription
                                logger.info("Completed chunk %d/%d", idx + 1, len(chunks))
                            except Exception as e:
                                logger.error("Error processing chunk %d: %s", index + 1, e)
                                sys.exit(1)

                    all_transcriptions = [results[i] for i in range(len(chunks))]
                    final_transcript = "\n".join(all_transcriptions)
                    _write_output(final_transcript, str(validated.output_file))
            else:
                final_transcript = transcription_backend.transcribe(audio_path)

                if validated.glossary_text and llm_backend is not None:
                    logger.info("Applying glossary correction...")
                    final_transcript = correct_with_glossary(
                        final_transcript, validated.glossary_text, llm_backend
                    )

                _write_output(final_transcript, str(validated.output_file))

        # Synthesis
        if validated.synthesise:
            output_stem = validated.output_file.with_suffix("")
            synthesis_output = str(output_stem) + "_synthesis.md"
            logger.info("Generating synthesis document...")
            if llm_backend is None:
                logger.error("LLM backend not configured")
                sys.exit(1)
            try:
                synthesis = synthesise_transcript(final_transcript, llm_backend)
                _write_output(synthesis, synthesis_output)
            except SynthesisError as e:
                logger.warning("%s", e)
                logger.warning(
                    "Transcript was saved successfully, but synthesis could not be generated."
                )

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

    args = parser.parse_args()

    try:
        validated = validate_cli_config(
            audio_file=args.audio_file,
            output_file=args.output_file,
            glossary=args.glossary,
            synthesise=args.synthesise,
            synthesise_only=getattr(args, "synthesise_only", False),
            parallel_workers=args.parallel_workers,
            local=args.local,
            model=args.model,
        )
        _run(validated)
    except ConfigurationError as e:
        for err in e.errors:
            logger.error("Error: %s", err)
        sys.exit(1)
    except TranscriberError as e:
        logger.error("Error: %s", e)
        sys.exit(1)
