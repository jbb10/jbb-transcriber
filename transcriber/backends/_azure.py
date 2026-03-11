"""Azure transcription and LLM backends using httpx.

Both backends are async-first and implement ``AsyncContextManager`` for
lifecycle management of the underlying ``httpx.AsyncClient``.
"""

from __future__ import annotations

import logging
import os
from types import TracebackType

import httpx

from transcriber._exceptions import LLMError, TranscriptionError
from transcriber._security import validate_https_url
from transcriber._settings import AzureLLMSettings, AzureTranscriptionSettings

logger = logging.getLogger(__name__)


class AzureTranscriptionBackend:
    """Transcription backend using Azure OpenAI.

    Args:
        settings: Azure transcription settings.
        client: Optional pre-configured httpx client.  If ``None``, one is
            created (and owned) by this backend.
    """

    def __init__(
        self,
        settings: AzureTranscriptionSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        validate_https_url(settings.api_url, name="Azure transcription URL")
        self._settings = settings
        self._client = client or httpx.AsyncClient(timeout=settings.request_timeout)
        self._owns_client = client is None

    async def transcribe(self, audio_path: str, *, time_offset: int = 0) -> str:
        """Transcribe an audio file via the Azure OpenAI API.

        Args:
            audio_path: Path to the audio file.
            time_offset: Seconds added to all timestamps.

        Returns:
            Formatted transcription text with speaker labels and timestamps.

        Raises:
            TranscriptionError: On API or I/O failure.
        """
        headers = {"api-key": self._settings.api_key}
        data = {
            "model": self._settings.model,
            "response_format": "diarized_json",
            "chunking_strategy": "auto",
        }

        try:
            with open(audio_path, "rb") as audio_file:
                files = {
                    "file": (
                        os.path.basename(audio_path),
                        audio_file,
                        "application/octet-stream",
                    )
                }
                logger.debug("Sending to API for transcription")

                response = await self._client.post(
                    self._settings.api_url,
                    headers=headers,
                    files=files,
                    data=data,
                )
                response.raise_for_status()
                result = response.json()
        except httpx.TimeoutException:
            raise TranscriptionError(
                "Request timed out. The audio file may be too large."
            ) from None
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response.text else None
            status = e.response.status_code
            logger.error(
                "Transcription API request failed [HTTP %s]: %s",
                status,
                body[:200] if body else str(e),
            )
            raise TranscriptionError(
                f"API request failed: {e}",
                status_code=status,
                response_body=body,
            ) from e
        except httpx.HTTPError as e:
            raise TranscriptionError(f"API request failed: {e}") from e
        except OSError as e:
            raise TranscriptionError(f"Could not read audio file: {e}") from e

        return self._parse_transcription_response(result, time_offset)

    @staticmethod
    def _parse_transcription_response(result: dict[str, object], time_offset: int) -> str:
        """Parse the transcription API JSON response."""
        if "segments" in result:
            output_lines: list[str] = []
            segments = result["segments"]
            if not isinstance(segments, list):
                raise TranscriptionError(
                    "Unexpected API response format",
                    response_body=str(result)[:500],
                )
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                speaker = segment.get("speaker", "Unknown")
                text = segment.get("text", "")
                start_val = segment.get("start", 0)
                end_val = segment.get("end", 0)
                start = (float(start_val) if start_val is not None else 0.0) + time_offset  # type: ignore[arg-type]
                end = (float(end_val) if end_val is not None else 0.0) + time_offset  # type: ignore[arg-type]
                output_lines.append(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}")
            return "\n".join(output_lines)
        elif "text" in result:
            return str(result["text"])
        else:
            raise TranscriptionError(
                "Unexpected API response format",
                response_body=str(result)[:500],
            )

    async def aclose(self) -> None:
        """Close the underlying HTTP client if owned by this backend."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AzureTranscriptionBackend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class AzureLLMBackend:
    """LLM backend using Azure OpenAI chat completions.

    Args:
        settings: Azure LLM settings.
        client: Optional pre-configured httpx client.  If ``None``, one is
            created (and owned) by this backend.
    """

    def __init__(
        self,
        settings: AzureLLMSettings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        validate_https_url(settings.api_url, name="Azure LLM URL")
        self._settings = settings
        self._client = client or httpx.AsyncClient(timeout=settings.request_timeout)
        self._owns_client = client is None

    async def complete(self, prompt: str, *, temperature: float = 0.1) -> str:
        """Send a prompt and return the LLM completion text.

        Args:
            prompt: Full prompt string.
            temperature: Sampling temperature.

        Returns:
            The completion text.

        Raises:
            LLMError: If the request fails.
        """
        headers = {
            "api-key": self._settings.api_key,
            "Content-Type": "application/json",
        }
        data = {
            "model": self._settings.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        try:
            response = await self._client.post(
                self._settings.api_url,
                headers=headers,
                json=data,
            )
            response.raise_for_status()
            result = response.json()
        except httpx.TimeoutException:
            raise LLMError("LLM request timed out") from None
        except httpx.HTTPStatusError as e:
            body = e.response.text[:500] if e.response.text else None
            status = e.response.status_code
            logger.error(
                "LLM API request failed [HTTP %s]: %s",
                status,
                body[:200] if body else str(e),
            )
            raise LLMError(
                f"LLM request failed: {e}",
                status_code=status,
                response_body=body,
            ) from e
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request failed: {e}") from e

        return self._parse_completion_response(result)

    @staticmethod
    def _parse_completion_response(result: dict[str, object]) -> str:
        """Parse the chat completions API JSON response."""
        if "choices" in result:
            choices = result["choices"]
            if isinstance(choices, list) and len(choices) > 0:
                choice = choices[0]
                if isinstance(choice, dict):
                    message = choice.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", "")
                        if isinstance(content, str) and content.strip():
                            return content.strip()

        raise LLMError(
            "Unexpected response format from LLM API",
            response_body=str(result)[:500],
        )

    async def aclose(self) -> None:
        """Close the underlying HTTP client if owned by this backend."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AzureLLMBackend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


# ---------------------------------------------------------------------------
# Backward-compatible constructors (accept raw key/url like the old API)
# ---------------------------------------------------------------------------

def create_azure_transcription_backend(
    api_key: str,
    api_url: str,
    *,
    request_timeout: int = 600,
) -> AzureTranscriptionBackend:
    """Create an ``AzureTranscriptionBackend`` from raw credentials."""
    settings = AzureTranscriptionSettings(
        api_key=api_key, api_url=api_url, request_timeout=request_timeout
    )
    return AzureTranscriptionBackend(settings)


def create_azure_llm_backend(
    api_key: str,
    api_url: str,
    *,
    request_timeout: int = 300,
) -> AzureLLMBackend:
    """Create an ``AzureLLMBackend`` from raw credentials."""
    settings = AzureLLMSettings(api_key=api_key, api_url=api_url, request_timeout=request_timeout)
    return AzureLLMBackend(settings)
