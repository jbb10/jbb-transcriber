"""Azure transcription and LLM backends using the OpenAI Python SDK.

Both backends are async-first and implement ``AsyncContextManager`` for
lifecycle management of the underlying ``AsyncOpenAI`` client.
"""

from __future__ import annotations

import logging
import os
from types import TracebackType
from typing import Any, cast

import openai
from openai.types.chat import ChatCompletion

from transcriber._exceptions import LLMError, TranscriptionError
from transcriber._security import validate_https_url
from transcriber._settings import AzureLLMSettings, AzureTranscriptionSettings

logger = logging.getLogger(__name__)


class AzureTranscriptionBackend:
    """Transcription backend using the OpenAI SDK pointed at a LiteLLM proxy.

    Args:
        settings: Azure transcription settings.
        client: Optional pre-configured AsyncOpenAI client.  If ``None``, one is
            created (and owned) by this backend.
    """

    def __init__(
        self,
        settings: AzureTranscriptionSettings,
        *,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        validate_https_url(settings.api_url, name="Azure transcription URL")
        self._settings = settings
        self._client = client or openai.AsyncOpenAI(
            base_url=settings.api_url,
            api_key=settings.api_key,
            timeout=settings.request_timeout,
        )
        self._owns_client = client is None

    async def transcribe(self, audio_path: str, *, time_offset: int = 0) -> str:
        """Transcribe an audio file via the LiteLLM proxy.

        Args:
            audio_path: Path to the audio file.
            time_offset: Seconds added to all timestamps.

        Returns:
            Formatted transcription text with speaker labels and timestamps.

        Raises:
            TranscriptionError: On API or I/O failure.
        """
        try:
            with open(audio_path, "rb") as audio_file:
                file_size_mb = os.path.getsize(audio_path) / 1_048_576
                logger.info(
                    "Transcribing audio (%.1f MB) — this may take a minute...",
                    file_size_mb,
                )
                result = await self._client.audio.transcriptions.create(  # type: ignore[call-overload]
                    model=self._settings.model,
                    file=(os.path.basename(audio_path), audio_file, "application/octet-stream"),
                    response_format="diarized_json",  # pyright: ignore[reportArgumentType]
                    chunking_strategy="auto",
                )
        except openai.APITimeoutError:
            raise TranscriptionError(
                "Request timed out. The audio file may be too large."
            ) from None
        except openai.APIStatusError as e:
            body: str | None = e.response.text[:500] if e.response.text else None
            status = e.status_code
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
        except openai.APIConnectionError as e:
            raise TranscriptionError(f"API request failed: {e}") from e
        except OSError as e:
            raise TranscriptionError(f"Could not read audio file: {e}") from e

        raw: dict[str, Any] = cast("dict[str, Any]", result.model_dump())  # type: ignore[union-attr]
        return self._parse_transcription_response(raw, time_offset)

    @staticmethod
    def _parse_transcription_response(result: dict[str, Any], time_offset: int) -> str:
        """Parse the transcription API JSON response."""
        if "segments" in result:
            output_lines: list[str] = []
            segments = result["segments"]
            if not isinstance(segments, list):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TranscriptionError(
                    "Unexpected API response format",
                    response_body=str(result)[:500],
                )
            for raw_segment in segments:  # pyright: ignore[reportUnknownVariableType]
                if not isinstance(raw_segment, dict):
                    continue
                segment: dict[str, Any] = cast("dict[str, Any]", raw_segment)
                speaker = str(segment.get("speaker", "Unknown"))
                text = str(segment.get("text", ""))
                start_val: Any = segment.get("start", 0)
                end_val: Any = segment.get("end", 0)
                start = float(start_val if start_val is not None else 0) + time_offset
                end = float(end_val if end_val is not None else 0) + time_offset
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
        """Close the underlying OpenAI client if owned by this backend."""
        if self._owns_client:
            await self._client.close()

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
    """LLM backend using the OpenAI SDK chat completions.

    Args:
        settings: Azure LLM settings.
        client: Optional pre-configured AsyncOpenAI client.  If ``None``, one is
            created (and owned) by this backend.
    """

    def __init__(
        self,
        settings: AzureLLMSettings,
        *,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        validate_https_url(settings.api_url, name="Azure LLM URL")
        self._settings = settings
        self._client = client or openai.AsyncOpenAI(
            base_url=settings.api_url,
            api_key=settings.api_key,
            timeout=settings.request_timeout,
        )
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
        try:
            result = await self._client.chat.completions.create(
                model=self._settings.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
        except openai.APITimeoutError:
            raise LLMError("LLM request timed out") from None
        except openai.APIStatusError as e:
            body: str | None = e.response.text[:500] if e.response.text else None
            status = e.status_code
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
        except openai.APIConnectionError as e:
            raise LLMError(f"LLM request failed: {e}") from e

        return self._parse_completion_response(result)

    @staticmethod
    def _parse_completion_response(result: ChatCompletion) -> str:
        """Parse the chat completions API response."""
        if result.choices and result.choices[0].message.content:
            content = result.choices[0].message.content.strip()
            if content:
                return content
        raise LLMError(
            "Unexpected response format from LLM API",
            response_body=str(result)[:500],
        )

    async def aclose(self) -> None:
        """Close the underlying OpenAI client if owned by this backend."""
        if self._owns_client:
            await self._client.close()

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
    model: str,
    request_timeout: int = 600,
) -> AzureTranscriptionBackend:
    """Create an ``AzureTranscriptionBackend`` from raw credentials."""
    settings = AzureTranscriptionSettings(
        api_key=api_key, api_url=api_url, model=model, request_timeout=request_timeout
    )
    return AzureTranscriptionBackend(settings)


def create_azure_llm_backend(
    api_key: str,
    api_url: str,
    *,
    model: str,
    request_timeout: int = 300,
) -> AzureLLMBackend:
    """Create an ``AzureLLMBackend`` from raw credentials."""
    settings = AzureLLMSettings(
        api_key=api_key,
        api_url=api_url,
        model=model,
        request_timeout=request_timeout,
    )
    return AzureLLMBackend(settings)
