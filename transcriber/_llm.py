"""LLM backends for glossary correction and synthesis.

Provides the AzureLLMBackend implementation and prompt-building helpers.
"""

from __future__ import annotations

import logging
import os
from importlib.resources import files as pkg_files
from pathlib import Path

import requests

from transcriber._exceptions import ConfigurationError, SynthesisError, TranscriptionError
from transcriber._protocols import LLMBackend
from transcriber._retry import retry_with_backoff

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt helpers (pure functions — no I/O, trivially testable)
# ---------------------------------------------------------------------------


def _load_prompt(filename: str) -> str:
    """Load a prompt template bundled with the package.

    Args:
        filename: Name of the prompt file (e.g. ``"correction_prompt.md"``).

    Returns:
        The prompt template text.
    """
    try:
        return pkg_files("transcriber").joinpath(filename).read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError):
        # Fallback for editable installs / development
        prompt_path = Path(__file__).parent / filename
        return prompt_path.read_text(encoding="utf-8")


def build_correction_prompt(transcript: str, glossary_text: str) -> str:
    """Build a glossary-correction prompt from a transcript and glossary.

    Args:
        transcript: The raw transcription text.
        glossary_text: The glossary content.

    Returns:
        Complete prompt string ready to send to an LLM.
    """
    template = _load_prompt("correction_prompt.md")
    return template.replace("{{glossary}}", glossary_text).replace("{{transcript}}", transcript)


def build_synthesis_prompt(transcript: str) -> str:
    """Build a synthesis prompt from a transcript.

    Args:
        transcript: The transcription text.

    Returns:
        Complete prompt string ready to send to an LLM.
    """
    template = _load_prompt("synthesis_prompt.md")
    return template.replace("{{transcript}}", transcript)


# ---------------------------------------------------------------------------
# Azure LLM backend
# ---------------------------------------------------------------------------


class AzureLLMBackend:
    """LLM backend using Azure OpenAI chat completions.

    Args:
        api_key: Azure API key for the text/chat model.
        api_url: Azure API endpoint URL for the text/chat model.
    """

    def __init__(self, api_key: str, api_url: str) -> None:
        self.api_key = api_key
        self.api_url = api_url

    @classmethod
    def from_env(cls) -> AzureLLMBackend:
        """Create an instance from environment variables.

        Raises:
            ConfigurationError: If required environment variables are missing.
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
        return cls(api_key, api_url)  # type: ignore[arg-type]

    def complete(self, prompt: str, *, temperature: float = 0.1, max_retries: int = 3) -> str:
        """Send a prompt and return the LLM completion text.

        Args:
            prompt: Full prompt string.
            temperature: Sampling temperature.
            max_retries: Number of retry attempts.

        Returns:
            The completion text.

        Raises:
            TranscriptionError: If the request fails after all retries.
        """
        headers = {"api-key": self.api_key, "Content-Type": "application/json"}
        data = {
            "model": "gpt-5.1",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        def _call() -> str:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=300,
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
                if content.strip():
                    return content.strip()

            raise TranscriptionError("Unexpected response format from LLM API")

        try:
            return retry_with_backoff(
                _call,
                max_retries=max_retries,
                exceptions=(requests.exceptions.RequestException, TranscriptionError),
                operation_name="LLM completion",
            )
        except requests.exceptions.Timeout:
            raise TranscriptionError("LLM request timed out") from None
        except requests.exceptions.RequestException as e:
            raise TranscriptionError(f"LLM request failed: {e}") from e


# ---------------------------------------------------------------------------
# High-level correction / synthesis helpers
# ---------------------------------------------------------------------------


def correct_with_glossary(
    transcript: str,
    glossary_text: str,
    llm: LLMBackend,
    *,
    max_retries: int = 3,
) -> str:
    """Correct a transcript using an LLM and glossary.

    On failure, falls back to the original uncorrected transcript.

    Args:
        transcript: Raw transcription text.
        glossary_text: Glossary content.
        llm: An LLM backend instance.
        max_retries: Number of retry attempts.

    Returns:
        Corrected transcript, or the original on failure.
    """
    prompt = build_correction_prompt(transcript, glossary_text)
    try:
        return llm.complete(prompt, temperature=0.1, max_retries=max_retries)
    except (TranscriptionError, Exception):
        logger.warning(
            "Glossary correction failed after %d attempts. Falling back to uncorrected transcript.",
            max_retries,
        )
        return transcript


def synthesise_transcript(
    transcript: str,
    llm: LLMBackend,
    *,
    max_retries: int = 3,
) -> str:
    """Generate a synthesis document from a transcript.

    Args:
        transcript: Transcription text.
        llm: An LLM backend instance.
        max_retries: Number of retry attempts.

    Returns:
        Synthesis markdown document.

    Raises:
        SynthesisError: If synthesis fails after all retries.
    """
    prompt = build_synthesis_prompt(transcript)
    try:
        return llm.complete(prompt, temperature=0.3, max_retries=max_retries)
    except (TranscriptionError, Exception) as e:
        raise SynthesisError(f"Synthesis failed after {max_retries} attempts: {e}") from e
