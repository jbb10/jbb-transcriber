"""Transcription backends: Azure OpenAI and local Whisper.

Each backend implements the TranscriptionBackend protocol from _protocols.py.
"""

from __future__ import annotations

import logging
import os
from types import ModuleType

import requests

from transcriber._exceptions import ConfigurationError, TranscriptionError
from transcriber._retry import retry_with_backoff

logger = logging.getLogger(__name__)


def format_whisper_output(segments: list[dict[str, object]], time_offset: int = 0) -> str:
    """Format Whisper transcription segments into timestamped text.

    Args:
        segments: List of Whisper segment dicts with ``start``, ``end``, ``text`` keys.
        time_offset: Offset in seconds to add to all timestamps (for chunked files).

    Returns:
        Formatted transcription with timestamps.
    """
    output_lines: list[str] = []
    for segment in segments:
        start_val = segment.get("start", 0)
        end_val = segment.get("end", 0)
        start = float(start_val) if start_val is not None else 0.0  # type: ignore[arg-type]
        end = float(end_val) if end_val is not None else 0.0  # type: ignore[arg-type]
        start += time_offset
        end += time_offset
        text = str(segment.get("text", "")).strip()
        if text:
            output_lines.append(f"[{start:.2f}s - {end:.2f}s] {text}")
    return "\n".join(output_lines)


class AzureTranscriptionBackend:
    """Transcription backend using Azure OpenAI's gpt-4o-transcribe-diarize.

    Args:
        api_key: Azure API key.
        api_url: Azure API endpoint URL.
        request_timeout: HTTP request timeout in seconds.
        max_retries: Maximum number of attempts (1 = no retry).
        base_delay: Base delay in seconds for exponential backoff between retries.
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        *,
        request_timeout: int = 600,
        max_retries: int = 1,
        base_delay: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.base_delay = base_delay

    @classmethod
    def from_env(
        cls,
        *,
        request_timeout: int = 600,
        max_retries: int = 1,
        base_delay: float = 2.0,
    ) -> AzureTranscriptionBackend:
        """Create an instance from environment variables.

        Args:
            request_timeout: HTTP request timeout in seconds.
            max_retries: Maximum number of attempts (1 = no retry).
            base_delay: Base delay in seconds for exponential backoff.

        Raises:
            ConfigurationError: If required environment variables are missing.
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
        return cls(
            api_key,  # type: ignore[arg-type]
            api_url,  # type: ignore[arg-type]
            request_timeout=request_timeout,
            max_retries=max_retries,
            base_delay=base_delay,
        )

    def transcribe(self, audio_path: str, *, time_offset: int = 0) -> str:
        """Transcribe an audio file via the Azure OpenAI API.

        Args:
            audio_path: Path to the audio file.
            time_offset: Seconds added to all timestamps.

        Returns:
            Formatted transcription text with speaker labels and timestamps.

        Raises:
            TranscriptionError: On API or I/O failure.
        """
        headers = {"api-key": self.api_key}
        data = {
            "model": "gpt-4o-transcribe-diarize",
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
        }

        def _call() -> str:
            with open(audio_path, "rb") as audio_file:
                files = {
                    "file": (
                        os.path.basename(audio_path),
                        audio_file,
                        "application/octet-stream",
                    )
                }
                logger.debug("Sending to API for transcription")

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                result = response.json()

                if "segments" in result:
                    output_lines: list[str] = []
                    for segment in result["segments"]:
                        speaker = segment.get("speaker", "Unknown")
                        text = segment.get("text", "")
                        start = segment.get("start", 0) + time_offset
                        end = segment.get("end", 0) + time_offset
                        output_lines.append(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}")
                    return "\n".join(output_lines)
                elif "text" in result:
                    return result["text"]
                else:
                    raise TranscriptionError(
                        "Unexpected API response format",
                        response_body=str(result),
                    )

        try:
            return retry_with_backoff(
                _call,
                max_retries=self.max_retries,
                base_delay=self.base_delay,
                exceptions=(requests.exceptions.RequestException,),
                operation_name="transcription",
            )
        except requests.exceptions.Timeout:
            raise TranscriptionError(
                "Request timed out. The audio file may be too large."
            ) from None
        except requests.exceptions.RequestException as e:
            body = None
            status = None
            if hasattr(e, "response") and e.response is not None:
                body = e.response.text
                status = e.response.status_code
            raise TranscriptionError(
                f"API request failed: {e}",
                status_code=status,
                response_body=body,
            ) from e
        except OSError as e:
            raise TranscriptionError(f"Could not read audio file: {e}") from e


def _import_whisper() -> ModuleType:
    """Lazily import the Whisper library.

    Raises:
        ConfigurationError: If openai-whisper is not installed.
    """
    try:
        import whisper  # type: ignore[import-not-found]

        return whisper
    except ImportError:
        raise ConfigurationError(
            [
                "openai-whisper is not installed (required for local mode). "
                'Install with: uv tool install "transcriber[local]"'
            ]
        ) from None


class WhisperTranscriptionBackend:
    """Transcription backend using a local OpenAI Whisper model.

    Args:
        model_name: Whisper model name (e.g. ``"base"``, ``"medium"``).
        device: Compute device (``"cpu"`` or ``"cuda"``).  Auto-detected if
            ``None``.
    """

    def __init__(self, model_name: str = "base", device: str | None = None) -> None:
        self.model_name = model_name
        if device is None:
            try:
                import torch  # type: ignore[import-not-found]

                self.device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[union-attr,unknown-member-type]
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

    def transcribe(self, audio_path: str, *, time_offset: int = 0) -> str:
        """Transcribe audio using a local Whisper model.

        Args:
            audio_path: Path to the audio file.
            time_offset: Seconds added to all timestamps.

        Returns:
            Formatted transcription text with timestamps.

        Raises:
            TranscriptionError: If transcription fails.
            ConfigurationError: If whisper is not installed.
        """
        whisper = _import_whisper()

        logger.info("Loading Whisper model '%s'", self.model_name)
        logger.debug("Using device: %s", self.device)

        model = whisper.load_model(self.model_name, device=self.device)
        logger.debug("Model loaded successfully")

        logger.info("Transcribing with local Whisper model")
        logger.debug("Language: auto-detect")

        result = model.transcribe(audio_path)

        if "language" in result:
            logger.debug("Detected language: %s", result["language"])

        segments = result.get("segments", [])
        if not segments:
            return result.get("text", "")

        return format_whisper_output(segments, time_offset)
