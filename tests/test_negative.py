"""
Negative tests for error handling (these don't require Azure credentials).
"""

import argparse
import builtins
import os
import tempfile

import pytest

from transcriber import (
    get_config,
    import_whisper,
    positive_int,
    transcribe_audio,
    validate_audio_file,
    validate_config,
)


class TestPositiveIntType:
    """Tests for positive_int argparse type."""

    def test_valid_positive_int(self):
        """Accepts positive integers."""
        assert positive_int("1") == 1
        assert positive_int("15") == 15
        assert positive_int("100") == 100

    def test_zero_raises_error(self):
        """Rejects zero."""
        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            positive_int("0")
        assert "at least 1" in str(exc_info.value)

    def test_negative_raises_error(self):
        """Rejects negative integers."""
        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            positive_int("-1")
        assert "at least 1" in str(exc_info.value)

    def test_non_integer_raises_error(self):
        """Rejects non-integer values."""
        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            positive_int("abc")
        assert "invalid int" in str(exc_info.value)


class TestValidateConfig:
    """Tests for validate_config function."""

    @pytest.fixture
    def valid_audio_file(self, short_audio_file):
        """Return a valid audio file path."""
        return short_audio_file

    @pytest.fixture
    def mock_args(self, valid_audio_file):
        """Create a mock args namespace with valid defaults."""
        return argparse.Namespace(
            audio_file=valid_audio_file,
            output_file=None,
            glossary=None,
            synthesise=False,
            parallel_workers=15,
            local=False,
            model="base",
            language=None,
        )

    def test_parallel_workers_too_high(self, mock_args, monkeypatch, capsys):
        """Rejects parallel_workers > 100."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")
        mock_args.parallel_workers = 200

        with pytest.raises(SystemExit) as exc_info:
            validate_config(mock_args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "cannot exceed 100" in captured.err

    def test_output_dir_not_exists(self, mock_args, monkeypatch, capsys):
        """Rejects output path in nonexistent directory."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")
        mock_args.output_file = "/nonexistent/directory/output.txt"

        with pytest.raises(SystemExit) as exc_info:
            validate_config(mock_args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Output directory does not exist" in captured.err

    def test_audio_file_not_found(self, mock_args, monkeypatch, capsys):
        """Rejects nonexistent audio file."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")
        mock_args.audio_file = "/nonexistent/audio.mp3"

        with pytest.raises(SystemExit) as exc_info:
            validate_config(mock_args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Audio file not found" in captured.err

    def test_glossary_file_not_found(self, mock_args, monkeypatch, capsys):
        """Rejects nonexistent glossary file."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")
        monkeypatch.setenv("AZURE_TEXT_API_KEY", "test-text-key")
        monkeypatch.setenv("AZURE_TEXT_URL", "https://test.example.com")
        mock_args.glossary = "/nonexistent/glossary.txt"

        with pytest.raises(SystemExit) as exc_info:
            validate_config(mock_args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Glossary file not found" in captured.err

    def test_multiple_errors_shown(self, mock_args, monkeypatch, capsys):
        """All validation errors are reported at once."""
        # Clear all env vars to trigger multiple errors
        monkeypatch.delenv("AZURE_TRANSCRIBE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TRANSCRIBE_URL", raising=False)
        mock_args.audio_file = "/nonexistent/audio.mp3"
        mock_args.parallel_workers = 200

        with pytest.raises(SystemExit) as exc_info:
            validate_config(mock_args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        # Should see multiple error messages
        assert "cannot exceed 100" in captured.err
        assert "Audio file not found" in captured.err
        assert "AZURE_TRANSCRIBE_API_KEY" in captured.err

    def test_synthesise_requires_text_api(self, mock_args, monkeypatch, capsys):
        """--synthesise requires text API credentials."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")
        # Ensure text API credentials are NOT set
        monkeypatch.delenv("AZURE_TEXT_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TEXT_URL", raising=False)
        mock_args.synthesise = True

        with pytest.raises(SystemExit) as exc_info:
            validate_config(mock_args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "AZURE_TEXT_API_KEY" in captured.err
        assert "--synthesise" in captured.err

    def test_valid_config_returns_dataclass(self, mock_args, monkeypatch):
        """Valid config returns ValidatedConfig dataclass."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")

        config = validate_config(mock_args)

        assert config.parallel_workers == 15
        assert config.synthesise is False
        assert config.glossary is None
        assert config.transcribe_api_key == "test-key"


class TestValidationHappensBeforeWork:
    """Tests that parameter validation occurs BEFORE any FFmpeg/conversion work."""

    def test_invalid_param_fails_before_conversion(self, short_audio_file, capsys):
        """Invalid parameters should fail BEFORE any conversion/FFmpeg work starts."""
        import subprocess
        import sys

        # Use a valid audio file but invalid parallel-workers (too high)
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "transcriber",
                short_audio_file,
                "--parallel-workers",
                "200",  # Invalid: exceeds 100
            ],
            capture_output=True,
            text=True,
            timeout=10,  # Should fail fast, not wait for conversion
        )

        assert result.returncode != 0
        # Should see the validation error
        assert "cannot exceed 100" in result.stderr
        # Should NOT see any conversion messages (validation failed first)
        assert "Converting" not in result.stderr
        assert "extracting audio" not in result.stderr.lower()

    def test_missing_glossary_fails_before_conversion(self, short_audio_file):
        """Missing glossary file should fail BEFORE any conversion work."""
        import os
        import subprocess
        import sys

        env = os.environ.copy()
        env["AZURE_TRANSCRIBE_API_KEY"] = "test-key"
        env["AZURE_TRANSCRIBE_URL"] = "https://test.example.com"
        env["AZURE_TEXT_API_KEY"] = "test-key"
        env["AZURE_TEXT_URL"] = "https://test.example.com"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "transcriber",
                short_audio_file,
                "--glossary",
                "/nonexistent/glossary.txt",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert "Glossary file not found" in result.stderr
        # Should NOT see conversion messages
        assert "Converting" not in result.stderr

    def test_missing_env_vars_fails_before_conversion(self, short_audio_file):
        """Missing environment variables should fail BEFORE any conversion work."""
        import subprocess
        import sys

        # Run with no Azure credentials set
        env = {"PATH": os.environ.get("PATH", "")}

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "transcriber",
                short_audio_file,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode != 0
        assert "AZURE_TRANSCRIBE_API_KEY" in result.stderr
        # Should NOT see conversion messages
        assert "Converting" not in result.stderr


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


class TestWhisperImport:
    """Tests for Whisper lazy import."""

    def test_import_whisper_missing_exits(self, monkeypatch, capsys):
        """import_whisper exits with helpful message when whisper not installed."""
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "whisper":
                raise ImportError("No module named 'whisper'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(SystemExit) as exc_info:
            import_whisper()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "openai-whisper" in captured.err
        assert "uv tool install" in captured.err
