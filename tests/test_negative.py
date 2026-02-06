"""
Negative tests for error handling (these don't require Azure credentials).
"""

import os
import tempfile

import pytest

from transcriber import get_config, transcribe_audio, validate_audio_file


class TestMissingCredentials:
    """Tests for missing Azure credentials."""

    def test_missing_api_key(self, clean_env):
        """Raises SystemExit when AZURE_TRANSCRIBE_API_KEY is missing."""
        with pytest.raises(SystemExit) as exc_info:
            get_config()

        assert exc_info.value.code == 1

    def test_missing_api_url(self, clean_env, monkeypatch):
        """Raises SystemExit when AZURE_TRANSCRIBE_URL is missing."""
        # Set the key but not the URL
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")

        with pytest.raises(SystemExit) as exc_info:
            get_config()

        assert exc_info.value.code == 1

    def test_missing_text_api_key_when_required(self, clean_env, monkeypatch):
        """Raises SystemExit when text API credentials are required but missing."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")

        with pytest.raises(SystemExit) as exc_info:
            get_config(require_text_api=True)

        assert exc_info.value.code == 1

    def test_missing_text_api_url_when_required(self, clean_env, monkeypatch):
        """Raises SystemExit when text API URL is required but missing."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")
        monkeypatch.setenv("AZURE_TEXT_API_KEY", "test-text-key")

        with pytest.raises(SystemExit) as exc_info:
            get_config(require_text_api=True)

        assert exc_info.value.code == 1


class TestFileValidation:
    """Tests for file validation errors."""

    def test_file_not_found(self):
        """Raises SystemExit for nonexistent audio file."""
        with pytest.raises(SystemExit) as exc_info:
            validate_audio_file("/nonexistent/path/to/audio.mp3")

        assert exc_info.value.code == 1

    def test_validates_existing_file(self, short_audio_file):
        """Does not raise for existing audio file."""
        # Should not raise any exception
        validate_audio_file(short_audio_file)

    def test_unrecognized_format_logs_warning(self, capsys):
        """Logs warning for unrecognized file format but doesn't fail."""
        # Create a file with unusual extension
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            # Should not raise (just logs a warning)
            validate_audio_file(temp_path)

            # Check stderr for warning message
            captured = capsys.readouterr()
            assert "not recognized" in captured.err.lower() or "attempting" in captured.err.lower()
        finally:
            os.unlink(temp_path)


class TestInvalidApiCredentials:
    """Tests for invalid API credentials (requires network)."""

    def test_invalid_api_key_returns_error(self, short_audio_file):
        """Invalid API key results in API error."""
        with pytest.raises(SystemExit):
            transcribe_audio(
                short_audio_file,
                api_key="invalid-api-key-12345",
                api_url="https://invalid-endpoint.openai.azure.com/openai/deployments/test/audio/transcriptions?api-version=2025-03-01-preview",
            )

    def test_invalid_url_returns_error(self, short_audio_file):
        """Invalid API URL results in request error."""
        with pytest.raises(SystemExit):
            transcribe_audio(
                short_audio_file,
                api_key="test-key",
                api_url="https://this-endpoint-does-not-exist-12345.openai.azure.com/test",
            )


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_audio_file(self):
        """Empty audio file is handled gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            # Write empty content (0 bytes)
            temp_path = f.name

        try:
            # Should fail when trying to get duration or process
            from transcriber import get_audio_duration

            duration = get_audio_duration(temp_path)
            # An empty file should return None or fail gracefully
            assert duration is None or duration == 0
        finally:
            os.unlink(temp_path)

    def test_corrupted_audio_file(self):
        """Corrupted audio file is handled gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            # Write garbage data
            f.write(b"This is not valid MP3 data, just random bytes: " + os.urandom(1000))
            temp_path = f.name

        try:
            from transcriber import get_audio_duration

            duration = get_audio_duration(temp_path)
            # Corrupted file should return None
            assert duration is None
        finally:
            os.unlink(temp_path)
