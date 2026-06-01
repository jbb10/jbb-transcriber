"""Configuration settings as plain dataclasses.

Each provider has its own settings type with a ``from_env()`` classmethod
for convenient environment-variable loading.  ``PipelineSettings`` holds
cross-cutting pipeline configuration (chunking, retry, concurrency).

Environment variable resolution
--------------------------------
Both backends share a single LiteLLM proxy endpoint.  Credentials are
resolved with per-app override → org-wide default fallback:

* ``JBB_TRANSCRIBER_API_KEY``      →  ``OPENAI_API_KEY``   (API key)
* ``JBB_TRANSCRIBER_BASE_URL``     →  ``OPENAI_BASE_URL``   (proxy base URL, e.g. ``/v1``)
* ``JBB_TRANSCRIBER_MODEL``                                 (transcription model name)
* ``JBB_TRANSCRIBER_TEXT_MODEL``                            (LLM/chat model name)
* ``JBB_TRANSCRIBER_REQUEST_TIMEOUT``                       (HTTP timeout in seconds, default 600)

Pipeline tuning (optional, override defaults without code changes):

* ``JBB_TRANSCRIBER_CHUNK_DURATION``   seconds per audio chunk (default 180 = 3 min)
* ``JBB_TRANSCRIBER_MAX_WORKERS``      max parallel chunk transcriptions (default 8)
* ``JBB_TRANSCRIBER_MAX_RETRIES``      max retry attempts per chunk (default 3)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from jbb_transcriber._exceptions import ConfigurationError
from jbb_transcriber._security import validate_https_url


def resolve_api_key() -> str | None:
    """Return JBB_TRANSCRIBER_API_KEY, falling back to OPENAI_API_KEY."""
    return os.getenv("JBB_TRANSCRIBER_API_KEY") or os.getenv("OPENAI_API_KEY") or None


def resolve_base_url() -> str | None:
    """Return JBB_TRANSCRIBER_BASE_URL, falling back to OPENAI_BASE_URL."""
    return os.getenv("JBB_TRANSCRIBER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None


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
          * api_key  — ``JBB_TRANSCRIBER_API_KEY`` → ``OPENAI_API_KEY``
          * api_url  — ``JBB_TRANSCRIBER_BASE_URL`` → ``OPENAI_BASE_URL``
          * model    — ``JBB_TRANSCRIBER_MODEL``

        Raises:
            ConfigurationError: If required env vars are missing.
            SecurityError: If the URL is not HTTPS.
        """
        errors: list[str] = []
        api_key = resolve_api_key()
        api_url = resolve_base_url()
        model = os.getenv("JBB_TRANSCRIBER_MODEL")

        if not api_key:
            errors.append(
                "Neither JBB_TRANSCRIBER_API_KEY nor OPENAI_API_KEY is set. "
                'Add to ~/.zshrc: export JBB_TRANSCRIBER_API_KEY="your-litellm-key"'
            )
        if not api_url:
            errors.append(
                "Neither JBB_TRANSCRIBER_BASE_URL nor OPENAI_BASE_URL is set. "
                'Add to ~/.zshrc: export JBB_TRANSCRIBER_BASE_URL="https://your-proxy.example.com/v1"'
            )
        if not model:
            errors.append(
                "JBB_TRANSCRIBER_MODEL environment variable is not set. "
                'Add to ~/.zshrc: export JBB_TRANSCRIBER_MODEL="your-model-name"'
            )
        if errors:
            raise ConfigurationError(errors)

        assert api_url is not None  # noqa: S101 — guarded above
        validate_https_url(api_url, name="JBB_TRANSCRIBER_BASE_URL")

        request_timeout_raw = os.getenv("JBB_TRANSCRIBER_REQUEST_TIMEOUT")
        request_timeout = int(request_timeout_raw) if request_timeout_raw else 600

        return cls(api_key=api_key, api_url=api_url, model=model, request_timeout=request_timeout)  # type: ignore[arg-type]


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
          * api_key  — ``JBB_TRANSCRIBER_API_KEY`` → ``OPENAI_API_KEY``
          * api_url  — ``JBB_TRANSCRIBER_BASE_URL`` → ``OPENAI_BASE_URL``
          * model    — ``JBB_TRANSCRIBER_TEXT_MODEL``

        Raises:
            ConfigurationError: If required env vars are missing.
            SecurityError: If the URL is not HTTPS.
        """
        errors: list[str] = []
        api_key = resolve_api_key()
        api_url = resolve_base_url()
        model = os.getenv("JBB_TRANSCRIBER_TEXT_MODEL")

        if not api_key:
            errors.append(
                "Neither JBB_TRANSCRIBER_API_KEY nor OPENAI_API_KEY is set. "
                'Add to ~/.zshrc: export JBB_TRANSCRIBER_API_KEY="your-litellm-key"'
            )
        if not api_url:
            errors.append(
                "Neither JBB_TRANSCRIBER_BASE_URL nor OPENAI_BASE_URL is set. "
                'Add to ~/.zshrc: export JBB_TRANSCRIBER_BASE_URL="https://your-proxy.example.com/v1"'
            )
        if not model:
            errors.append(
                "JBB_TRANSCRIBER_TEXT_MODEL environment variable is not set. "
                'Add to ~/.zshrc: export JBB_TRANSCRIBER_TEXT_MODEL="your-model-name"'
            )
        if errors:
            raise ConfigurationError(errors)

        assert api_url is not None  # noqa: S101 — guarded above
        validate_https_url(api_url, name="JBB_TRANSCRIBER_BASE_URL")

        return cls(api_key=api_key, api_url=api_url, model=model)  # type: ignore[arg-type]


@dataclass(frozen=True)
class WhisperSettings:
    """Settings for the local Whisper transcription backend."""

    model_name: str = "base"
    device: str | None = None  # auto-detect


@dataclass(frozen=True)
class PipelineSettings:
    """Cross-cutting pipeline configuration."""

    chunk_duration: int = 180
    parallel_workers: int = 8
    max_duration_before_split: int = 180
    max_retries: int = 3
    base_delay: float = 2.0
    max_glossary_size: int = 500_000
    fail_on_correction_error: bool = False

    @classmethod
    def from_env(cls) -> PipelineSettings:
        """Create settings from environment variables, falling back to defaults.

        Reads:
          * ``JBB_TRANSCRIBER_CHUNK_DURATION``  — seconds per audio chunk (default 300)
          * ``JBB_TRANSCRIBER_MAX_WORKERS``     — parallel chunk workers (default 2)
          * ``JBB_TRANSCRIBER_MAX_RETRIES``     — retry attempts per chunk (default 3)

        Raises:
            ValueError: If an env var is set but cannot be parsed as an integer.
        """
        chunk_duration_raw = os.getenv("JBB_TRANSCRIBER_CHUNK_DURATION")
        max_workers_raw = os.getenv("JBB_TRANSCRIBER_MAX_WORKERS")
        max_retries_raw = os.getenv("JBB_TRANSCRIBER_MAX_RETRIES")

        chunk_duration = int(chunk_duration_raw) if chunk_duration_raw else 180
        parallel_workers = int(max_workers_raw) if max_workers_raw else 8
        max_retries = int(max_retries_raw) if max_retries_raw else 3

        return cls(
            chunk_duration=chunk_duration,
            max_duration_before_split=chunk_duration,
            parallel_workers=parallel_workers,
            max_retries=max_retries,
        )
