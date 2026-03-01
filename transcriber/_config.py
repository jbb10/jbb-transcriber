"""Configuration validation for both CLI and library usage.

Provides ValidatedConfig for the CLI layer, and helper functions for
building backend instances from validated parameters.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from transcriber._audio import log_audio_file_info, probe_audio_stream
from transcriber._exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ValidatedConfig:
    """Validated configuration for a transcription job.

    All fields are validated before construction.  This dataclass is used
    internally by the CLI to carry validated state; library users normally
    interact with the high-level ``transcribe_file`` / ``synthesise_text``
    functions instead.
    """

    audio_file: Path
    output_file: Path
    glossary: Path | None
    glossary_text: str | None
    synthesise: bool
    synthesise_only: bool
    parallel_workers: int
    # Credential strings (may be empty in local / synthesise-only mode)
    transcribe_api_key: str
    transcribe_url: str
    text_api_key: str | None
    text_url: str | None
    # Local mode
    local_mode: bool
    whisper_model: str


def validate_cli_config(  # noqa: C901 — complexity is inherent to validation
    *,
    audio_file: str,
    output_file: str | None,
    glossary: str | None,
    synthesise: bool,
    synthesise_only: bool,
    parallel_workers: int,
    local: bool,
    model: str,
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
        local: Whether ``--local`` was passed.
        model: Whisper model name.

    Returns:
        A fully validated ``ValidatedConfig``.

    Raises:
        ConfigurationError: If any validation fails.
    """
    errors: list[str] = []

    # --- Conflicting flags ---
    if synthesise_only and synthesise:
        errors.append("Cannot use --synthesise and --synthesise-only together")

    # --- Paths ---
    audio_path = Path(audio_file)
    if output_file is None:
        out_path = audio_path.with_suffix(".txt")
    else:
        out_path = Path(output_file)

    # --- parallel_workers ---
    if parallel_workers > 100:
        errors.append(f"--parallel-workers cannot exceed 100, got {parallel_workers}")

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
        raise ConfigurationError(errors)

    # --- Load glossary text (validation passed) ---
    if glossary_path:
        try:
            glossary_text = glossary_path.read_text(encoding="utf-8")
            logger.info("Loaded glossary from: %s", glossary_path)
        except OSError as e:
            raise ConfigurationError([f"Could not read glossary file: {e}"]) from e

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
        transcribe_api_key=transcribe_api_key,  # type: ignore[arg-type]
        transcribe_url=transcribe_url,  # type: ignore[arg-type]
        text_api_key=text_api_key,
        text_url=text_url,
        local_mode=local,
        whisper_model=model,
    )
