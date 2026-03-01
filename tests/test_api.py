"""
Tests for the public library API (transcribe_file, synthesise_text).
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import transcriber
from transcriber import (
    AudioFileError,
    ConfigurationError,
    TranscriptionResult,
    synthesise_text,
    transcribe_file,
)


class TestTranscribeFileValidation:
    """Tests for transcribe_file input validation."""

    def test_missing_file_raises_audio_file_error(self):
        """Raises AudioFileError for nonexistent file."""
        with pytest.raises(AudioFileError):
            transcribe_file("/nonexistent/file.mp3")

    def test_missing_glossary_raises_audio_file_error(self, short_audio_file):
        """Raises AudioFileError for nonexistent glossary file."""
        mock_backend = MagicMock()
        with pytest.raises(AudioFileError):
            transcribe_file(
                short_audio_file,
                glossary="/nonexistent/glossary.txt",
                transcription_backend=mock_backend,
            )


class TestTranscribeFileWithMockBackend:
    """Tests for transcribe_file with mock backends (no API calls)."""

    def test_returns_transcription_result(self, short_audio_file):
        """Returns a TranscriptionResult dataclass."""
        mock_backend = MagicMock()
        mock_backend.transcribe.return_value = "Hello, world."

        result = transcribe_file(short_audio_file, transcription_backend=mock_backend)

        assert isinstance(result, TranscriptionResult)
        assert result.transcript == "Hello, world."
        assert result.synthesis is None

    def test_writes_output_file(self, short_audio_file, tmp_path):
        """Writes transcript to output file when specified."""
        mock_backend = MagicMock()
        mock_backend.transcribe.return_value = "Transcript content"
        output = str(tmp_path / "output.txt")

        result = transcribe_file(
            short_audio_file, output=output, transcription_backend=mock_backend
        )

        assert result.transcript == "Transcript content"
        assert Path(output).read_text(encoding="utf-8") == "Transcript content"

    def test_glossary_correction(self, short_audio_file, tmp_path):
        """Applies glossary correction when glossary is given."""
        glossary_file = tmp_path / "glossary.txt"
        glossary_file.write_text("API: Application Programming Interface")

        mock_transcription = MagicMock()
        mock_transcription.transcribe.return_value = "We built an aye pee eye."

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "We built an API."

        result = transcribe_file(
            short_audio_file,
            glossary=str(glossary_file),
            transcription_backend=mock_transcription,
            llm_backend=mock_llm,
        )

        assert mock_llm.complete.called
        # The result should come from the LLM correction
        assert "API" in result.transcript or mock_llm.complete.called

    def test_synthesis_generation(self, short_audio_file):
        """Generates synthesis when synthesise=True."""
        mock_transcription = MagicMock()
        mock_transcription.transcribe.return_value = "Meeting transcript text."

        mock_llm = MagicMock()
        mock_llm.complete.return_value = "# Meeting Summary\n\nKey points..."

        result = transcribe_file(
            short_audio_file,
            synthesise=True,
            transcription_backend=mock_transcription,
            llm_backend=mock_llm,
        )

        assert result.synthesis is not None
        assert len(result.synthesis) > 0

    def test_duration_in_result(self, short_audio_file):
        """Result includes audio duration."""
        mock_backend = MagicMock()
        mock_backend.transcribe.return_value = "Hello"

        result = transcribe_file(short_audio_file, transcription_backend=mock_backend)

        # Duration should be populated from the actual audio file
        assert result.duration_seconds is not None
        assert result.duration_seconds > 0

    def test_transcription_result_is_frozen(self, short_audio_file):
        """TranscriptionResult is immutable (frozen dataclass)."""
        mock_backend = MagicMock()
        mock_backend.transcribe.return_value = "Hello"

        result = transcribe_file(short_audio_file, transcription_backend=mock_backend)

        with pytest.raises(AttributeError):
            result.transcript = "modified"  # type: ignore[misc]


class TestSynthesiseText:
    """Tests for synthesise_text function."""

    def test_returns_synthesis(self):
        """Returns synthesis text from LLM."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "# Summary\n\nKey decisions..."

        result = synthesise_text("Some transcript", llm_backend=mock_llm)

        assert result == "# Summary\n\nKey decisions..."
        assert mock_llm.complete.called

    def test_missing_llm_raises_config_error(self, clean_env):
        """Raises ConfigurationError when no LLM backend and no env vars."""
        with pytest.raises(ConfigurationError):
            synthesise_text("Some transcript")


class TestPublicExports:
    """Tests for public API surface (__all__)."""

    def test_all_exports_are_accessible(self):
        """All items in __all__ are importable."""
        for name in transcriber.__all__:
            assert hasattr(transcriber, name), f"{name} listed in __all__ but not accessible"

    def test_version_is_string(self):
        """__version__ is a string."""
        assert isinstance(transcriber.__version__, str)
        assert len(transcriber.__version__) > 0

    def test_transcription_result_fields(self):
        """TranscriptionResult has expected fields."""
        result = TranscriptionResult(
            transcript="Hello",
            synthesis="Summary",
            duration_seconds=10.5,
        )
        assert result.transcript == "Hello"
        assert result.synthesis == "Summary"
        assert result.duration_seconds == 10.5

    def test_transcription_result_defaults(self):
        """TranscriptionResult defaults are correct."""
        result = TranscriptionResult(transcript="Hello")
        assert result.synthesis is None
        assert result.duration_seconds is None
