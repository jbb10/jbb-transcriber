"""Backend package — provider-specific transcription and LLM implementations.

All backends implement the protocols defined in ``jbb_transcriber._protocols``.
"""

from __future__ import annotations

from jbb_transcriber.backends._azure import (
    AzureLLMBackend,
    AzureTranscriptionBackend,
    create_azure_llm_backend,
    create_azure_transcription_backend,
)
from jbb_transcriber.backends._whisper import (
    WhisperTranscriptionBackend,
    format_whisper_output,
)

__all__ = [
    "AzureLLMBackend",
    "AzureTranscriptionBackend",
    "WhisperTranscriptionBackend",
    "create_azure_llm_backend",
    "create_azure_transcription_backend",
    "create_transcription_backend",
    "create_llm_backend",
    "format_whisper_output",
]


def create_transcription_backend(
    provider: str = "azure",
    **kwargs: object,
) -> AzureTranscriptionBackend | WhisperTranscriptionBackend:
    """Factory: create a transcription backend by provider name.

    Args:
        provider: ``"azure"`` or ``"whisper"``.
        **kwargs: Passed to the backend constructor.

    Returns:
        A transcription backend instance.

    Raises:
        ValueError: If the provider is unknown.
    """
    if provider == "azure":
        return create_azure_transcription_backend(**kwargs)  # type: ignore[arg-type]
    elif provider == "whisper":
        return WhisperTranscriptionBackend(**kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown transcription provider: {provider!r}")


def create_llm_backend(
    provider: str = "azure",
    **kwargs: object,
) -> AzureLLMBackend:
    """Factory: create an LLM backend by provider name.

    Args:
        provider: ``"azure"`` (only option currently).
        **kwargs: Passed to the backend constructor.

    Returns:
        An LLM backend instance.

    Raises:
        ValueError: If the provider is unknown.
    """
    if provider == "azure":
        return create_azure_llm_backend(**kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")
