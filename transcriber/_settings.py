"""Configuration settings as plain dataclasses.

Each provider has its own settings type with a ``from_env()`` classmethod
for convenient environment-variable loading.  ``PipelineSettings`` holds
cross-cutting pipeline configuration (chunking, retry, concurrency).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from transcriber._exceptions import ConfigurationError
from transcriber._security import validate_https_url


@dataclass(frozen=True)
class AzureTranscriptionSettings:
    """Settings for the Azure OpenAI transcription backend."""

    api_key: str
    api_url: str
    model: str = "gpt-4o-transcribe-diarize"
    request_timeout: int = 600

    @classmethod
    def from_env(cls) -> AzureTranscriptionSettings:
        """Create settings from environment variables.

        Raises:
            ConfigurationError: If required env vars are missing.
            SecurityError: If the URL is not HTTPS.
        """
        errors: list[str] = []
        api_key = os.getenv("AZURE_TRANSCRIBE_API_KEY")
        api_url = os.getenv("AZURE_TRANSCRIBE_URL")

        if not api_key:
            errors.append(
                "AZURE_TRANSCRIBE_API_KEY environment variable is not set. "
                'Add to ~/.zshrc: export AZURE_TRANSCRIBE_API_KEY="your-api-key"'
            )
        if not api_url:
            errors.append(
                "AZURE_TRANSCRIBE_URL environment variable is not set. "
                'Add to ~/.zshrc: export AZURE_TRANSCRIBE_URL="your-endpoint-url"'
            )
        if errors:
            raise ConfigurationError(errors)

        assert api_url is not None  # noqa: S101 — guarded above
        validate_https_url(api_url, name="AZURE_TRANSCRIBE_URL")

        return cls(api_key=api_key, api_url=api_url)  # type: ignore[arg-type]


@dataclass(frozen=True)
class AzureLLMSettings:
    """Settings for the Azure OpenAI LLM (chat-completions) backend."""

    api_key: str
    api_url: str
    model: str = "gpt-5.1"
    request_timeout: int = 300

    @classmethod
    def from_env(cls) -> AzureLLMSettings:
        """Create settings from environment variables.

        Raises:
            ConfigurationError: If required env vars are missing.
            SecurityError: If the URL is not HTTPS.
        """
        errors: list[str] = []
        api_key = os.getenv("AZURE_TEXT_API_KEY")
        api_url = os.getenv("AZURE_TEXT_URL")

        if not api_key:
            errors.append(
                "AZURE_TEXT_API_KEY environment variable is not set. "
                'Add to ~/.zshrc: export AZURE_TEXT_API_KEY="your-api-key"'
            )
        if not api_url:
            errors.append(
                "AZURE_TEXT_URL environment variable is not set. "
                'Add to ~/.zshrc: export AZURE_TEXT_URL="your-endpoint-url"'
            )
        if errors:
            raise ConfigurationError(errors)

        assert api_url is not None  # noqa: S101 — guarded above
        validate_https_url(api_url, name="AZURE_TEXT_URL")

        return cls(api_key=api_key, api_url=api_url)  # type: ignore[arg-type]


@dataclass(frozen=True)
class WhisperSettings:
    """Settings for the local Whisper transcription backend."""

    model_name: str = "base"
    device: str | None = None  # auto-detect


@dataclass(frozen=True)
class PipelineSettings:
    """Cross-cutting pipeline configuration."""

    chunk_duration: int = 900
    parallel_workers: int = 15
    max_duration_before_split: int = 1400
    max_retries: int = 3
    base_delay: float = 2.0
    max_glossary_size: int = 500_000
    fail_on_correction_error: bool = False
