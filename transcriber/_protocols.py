"""Protocol types for dependency injection.

These define the interfaces that transcription and LLM backends must implement.
Library users can provide custom implementations via these protocols.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TranscriptionBackend(Protocol):
    """Interface for audio transcription backends.

    Implementations must accept an audio file path and return formatted text.
    """

    def transcribe(self, audio_path: str, *, time_offset: int = 0) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file.
            time_offset: Seconds to add to all timestamps (for chunked files).

        Returns:
            Formatted transcription text.

        Raises:
            TranscriptionError: If transcription fails.
        """
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """Interface for LLM text completion backends.

    Used for glossary correction and synthesis generation.
    """

    def complete(self, prompt: str, *, temperature: float = 0.1, max_retries: int = 3) -> str:
        """Send a prompt to the LLM and return the completion.

        Args:
            prompt: The full prompt text.
            temperature: Sampling temperature.
            max_retries: Number of retry attempts with exponential backoff.

        Returns:
            The LLM's response text.

        Raises:
            TranscriptionError: If the request fails after all retries.
        """
        ...
