"""Configuration settings as plain dataclasses.

Each provider has its own settings type with a ``from_env()`` classmethod
for convenient environment-variable loading.  ``PipelineSettings`` holds
cross-cutting pipeline configuration (chunking, retry, concurrency).

Environment variable resolution
--------------------------------
Both backends share a single LiteLLM proxy endpoint.  Credentials are
resolved with per-app override → org-wide default fallback:

* ``TRANSCRIBER_API_KEY``  →  ``OPENAI_API_KEY``   (API key)
* ``TRANSCRIBER_BASE_URL`` →  ``OPENAI_BASE_URL``   (proxy base URL, e.g. ``/v1``)
* ``TRANSCRIBER_MODEL``                              (transcription model name)
* ``TRANSCRIBER_TEXT_MODEL``                         (LLM/chat model name)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from transcriber._exceptions import ConfigurationError
from transcriber._security import validate_https_url


def resolve_api_key() -> str | None:
    """Return TRANSCRIBER_API_KEY, falling back to OPENAI_API_KEY."""
    return os.getenv("TRANSCRIBER_API_KEY") or os.getenv("OPENAI_API_KEY") or None


def resolve_base_url() -> str | None:
    """Return TRANSCRIBER_BASE_URL, falling back to OPENAI_BASE_URL."""
    return os.getenv("TRANSCRIBER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None


@dataclass(frozen=True)
class AzureTranscriptionSettings:
    """Settings for the transcription backend (LiteLLM proxy via OpenAI SDK)."""

    api_key: str
    api_url: str
    model: str
    request_timeout: int = 600

    @classmethod
    def from_env(cls) -> AzureTranscriptionSettings:
        """Create settings from environment variables.

        Resolution order:
          * api_key  — ``TRANSCRIBER_API_KEY`` → ``OPENAI_API_KEY``
          * api_url  — ``TRANSCRIBER_BASE_URL`` → ``OPENAI_BASE_URL``
          * model    — ``TRANSCRIBER_MODEL``

        Raises:
            ConfigurationError: If required env vars are missing.
            SecurityError: If the URL is not HTTPS.
        """
        errors: list[str] = []
        api_key = resolve_api_key()
        api_url = resolve_base_url()
        model = os.getenv("TRANSCRIBER_MODEL")

        if not api_key:
            errors.append(
                "Neither TRANSCRIBER_API_KEY nor OPENAI_API_KEY is set. "
                'Add to ~/.zshrc: export TRANSCRIBER_API_KEY="your-litellm-key"'
            )
        if not api_url:
            errors.append(
                "Neither TRANSCRIBER_BASE_URL nor OPENAI_BASE_URL is set. "
                'Add to ~/.zshrc: export TRANSCRIBER_BASE_URL="https://your-proxy.example.com/v1"'
            )
        if not model:
            errors.append(
                "TRANSCRIBER_MODEL environment variable is not set. "
                'Add to ~/.zshrc: export TRANSCRIBER_MODEL="your-model-name"'
            )
        if errors:
            raise ConfigurationError(errors)

        assert api_url is not None  # noqa: S101 — guarded above
        validate_https_url(api_url, name="TRANSCRIBER_BASE_URL")

        return cls(api_key=api_key, api_url=api_url, model=model)  # type: ignore[arg-type]


@dataclass(frozen=True)
class AzureLLMSettings:
    """Settings for the LLM chat-completions backend (LiteLLM proxy via OpenAI SDK)."""

    api_key: str
    api_url: str
    model: str
    request_timeout: int = 300

    @classmethod
    def from_env(cls) -> AzureLLMSettings:
        """Create settings from environment variables.

        Resolution order:
          * api_key  — ``TRANSCRIBER_API_KEY`` → ``OPENAI_API_KEY``
          * api_url  — ``TRANSCRIBER_BASE_URL`` → ``OPENAI_BASE_URL``
          * model    — ``TRANSCRIBER_TEXT_MODEL``

        Raises:
            ConfigurationError: If required env vars are missing.
            SecurityError: If the URL is not HTTPS.
        """
        errors: list[str] = []
        api_key = resolve_api_key()
        api_url = resolve_base_url()
        model = os.getenv("TRANSCRIBER_TEXT_MODEL")

        if not api_key:
            errors.append(
                "Neither TRANSCRIBER_API_KEY nor OPENAI_API_KEY is set. "
                'Add to ~/.zshrc: export TRANSCRIBER_API_KEY="your-litellm-key"'
            )
        if not api_url:
            errors.append(
                "Neither TRANSCRIBER_BASE_URL nor OPENAI_BASE_URL is set. "
                'Add to ~/.zshrc: export TRANSCRIBER_BASE_URL="https://your-proxy.example.com/v1"'
            )
        if not model:
            errors.append(
                "TRANSCRIBER_TEXT_MODEL environment variable is not set. "
                'Add to ~/.zshrc: export TRANSCRIBER_TEXT_MODEL="your-model-name"'
            )
        if errors:
            raise ConfigurationError(errors)

        assert api_url is not None  # noqa: S101 — guarded above
        validate_https_url(api_url, name="TRANSCRIBER_BASE_URL")

        return cls(api_key=api_key, api_url=api_url, model=model)  # type: ignore[arg-type]


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
    max_duration_before_split: int = 900
    max_retries: int = 3
    base_delay: float = 2.0
    max_glossary_size: int = 500_000
    fail_on_correction_error: bool = False
