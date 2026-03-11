"""Local Whisper transcription backend.

Wraps the sync ``openai-whisper`` library in ``asyncio.to_thread()`` so it
integrates cleanly with the async pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from types import ModuleType

from transcriber._exceptions import ConfigurationError, TranscriptionError
from transcriber._settings import WhisperSettings

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
        settings: Whisper settings (model name and device).
    """

    def __init__(
        self,
        settings: WhisperSettings | None = None,
        *,
        model_name: str = "base",
        device: str | None = None,
    ) -> None:
        if settings is not None:
            self._model_name = settings.model_name
            device = settings.device
        else:
            self._model_name = model_name

        if device is None:
            try:
                import torch  # type: ignore[import-not-found]

                self._device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[union-attr,unknown-member-type]
            except ImportError:
                self._device = "cpu"
        else:
            self._device = device

    def _transcribe_sync(self, audio_path: str, time_offset: int = 0) -> str:
        """Sync transcription — called via ``asyncio.to_thread()``."""
        whisper = _import_whisper()

        logger.info("Loading Whisper model '%s'", self._model_name)
        logger.debug("Using device: %s", self._device)

        model = whisper.load_model(self._model_name, device=self._device)
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

    async def transcribe(self, audio_path: str, *, time_offset: int = 0) -> str:
        """Transcribe audio using a local Whisper model.

        The CPU-bound work is offloaded to a thread via ``asyncio.to_thread()``.

        Args:
            audio_path: Path to the audio file.
            time_offset: Seconds added to all timestamps.

        Returns:
            Formatted transcription text with timestamps.

        Raises:
            TranscriptionError: If transcription fails.
            ConfigurationError: If whisper is not installed.
        """
        try:
            return await asyncio.to_thread(self._transcribe_sync, audio_path, time_offset)
        except (ConfigurationError, TranscriptionError):
            raise
        except Exception as e:
            raise TranscriptionError(f"Whisper transcription failed: {e}") from e
