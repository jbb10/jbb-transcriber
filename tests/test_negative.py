"""
Negative tests for error handling (these don't require Azure credentials).
"""

import builtins
import os
import tempfile

import pytest

from transcriber import ConfigurationError
from transcriber.backends._whisper import format_whisper_output
from transcriber.cli import positive_int, validate_cli_config


class TestPositiveIntType:
    """Tests for positive_int argparse type."""

    def test_valid_positive_int(self):
        """Accepts positive integers."""
        assert positive_int("1") == 1
        assert positive_int("15") == 15
        assert positive_int("100") == 100

    def test_zero_raises_error(self):
        """Rejects zero."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            positive_int("0")
        assert "at least 1" in str(exc_info.value)

    def test_negative_raises_error(self):
        """Rejects negative integers."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            positive_int("-1")
        assert "at least 1" in str(exc_info.value)

    def test_non_integer_raises_error(self):
        """Rejects non-integer values."""
        import argparse

        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            positive_int("abc")
        assert "invalid int" in str(exc_info.value)


class TestValidateCliConfig:
    """Tests for validate_cli_config function."""

    @pytest.fixture
    def valid_audio_file(self, short_audio_file):
        """Return a valid audio file path."""
        return short_audio_file

    @pytest.fixture
    def valid_kwargs(self, valid_audio_file):
        """Create valid keyword args for validate_cli_config."""
        return {
            "audio_file": valid_audio_file,
            "output_file": None,
            "glossary": None,
            "synthesise": False,
            "synthesise_only": False,
            "parallel_workers": 15,
            "local": False,
            "model": "base",
        }

    def test_parallel_workers_too_high(self, valid_kwargs, monkeypatch):
        """Rejects parallel_workers > 100."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")
        valid_kwargs["parallel_workers"] = 200

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**valid_kwargs)

        assert any("cannot exceed 100" in e for e in exc_info.value.errors)

    def test_output_dir_not_exists(self, valid_kwargs, monkeypatch):
        """Rejects output path in nonexistent directory."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")
        valid_kwargs["output_file"] = "/nonexistent/directory/output.txt"

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**valid_kwargs)

        assert any("Output directory does not exist" in e for e in exc_info.value.errors)

    def test_audio_file_not_found(self, valid_kwargs, monkeypatch):
        """Rejects nonexistent audio file."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")
        valid_kwargs["audio_file"] = "/nonexistent/audio.mp3"

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**valid_kwargs)

        assert any("Audio file not found" in e for e in exc_info.value.errors)

    def test_glossary_file_not_found(self, valid_kwargs, monkeypatch):
        """Rejects nonexistent glossary file."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")
        monkeypatch.setenv("AZURE_TEXT_API_KEY", "test-text-key")
        monkeypatch.setenv("AZURE_TEXT_URL", "https://test.example.com")
        valid_kwargs["glossary"] = "/nonexistent/glossary.txt"

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**valid_kwargs)

        assert any("Glossary file not found" in e for e in exc_info.value.errors)

    def test_multiple_errors_shown(self, valid_kwargs, monkeypatch):
        """All validation errors are reported at once."""
        monkeypatch.delenv("AZURE_TRANSCRIBE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TRANSCRIBE_URL", raising=False)
        valid_kwargs["audio_file"] = "/nonexistent/audio.mp3"
        valid_kwargs["parallel_workers"] = 200

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**valid_kwargs)

        errors_text = " ".join(exc_info.value.errors)
        assert "cannot exceed 100" in errors_text
        assert "Audio file not found" in errors_text
        assert "AZURE_TRANSCRIBE_API_KEY" in errors_text

    def test_synthesise_requires_text_api(self, valid_kwargs, monkeypatch):
        """--synthesise requires text API credentials."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")
        monkeypatch.delenv("AZURE_TEXT_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TEXT_URL", raising=False)
        valid_kwargs["synthesise"] = True

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**valid_kwargs)

        errors_text = " ".join(exc_info.value.errors)
        assert "AZURE_TEXT_API_KEY" in errors_text
        assert "--synthesise" in errors_text

    def test_synthesise_only_requires_text_api(self, valid_kwargs, monkeypatch):
        """--synthesise-only requires text API credentials."""
        monkeypatch.delenv("AZURE_TRANSCRIBE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TRANSCRIBE_URL", raising=False)
        monkeypatch.delenv("AZURE_TEXT_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TEXT_URL", raising=False)
        valid_kwargs["synthesise_only"] = True

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**valid_kwargs)

        errors_text = " ".join(exc_info.value.errors)
        assert "AZURE_TEXT_API_KEY" in errors_text
        assert "--synthesise-only" in errors_text

    def test_synthesise_only_and_synthesise_conflict(self, valid_kwargs, monkeypatch):
        """--synthesise and --synthesise-only cannot be used together."""
        monkeypatch.setenv("AZURE_TEXT_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TEXT_URL", "https://test.example.com")
        valid_kwargs["synthesise"] = True
        valid_kwargs["synthesise_only"] = True

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**valid_kwargs)

        assert any(
            "Cannot use --synthesise and --synthesise-only together" in e
            for e in exc_info.value.errors
        )

    def test_synthesise_only_transcript_not_found(self, valid_kwargs, monkeypatch):
        """--synthesise-only fails if transcript file not found."""
        monkeypatch.setenv("AZURE_TEXT_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TEXT_URL", "https://test.example.com")
        valid_kwargs["synthesise_only"] = True
        valid_kwargs["audio_file"] = "/nonexistent/transcript.txt"

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**valid_kwargs)

        assert any("Transcript file not found" in e for e in exc_info.value.errors)

    def test_synthesise_only_skips_audio_validation(self, valid_kwargs, monkeypatch, tmp_path):
        """--synthesise-only skips audio stream validation (accepts text files)."""
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("Some transcript content")

        monkeypatch.setenv("AZURE_TEXT_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TEXT_URL", "https://test.example.com")
        valid_kwargs["synthesise_only"] = True
        valid_kwargs["audio_file"] = str(transcript)

        config = validate_cli_config(**valid_kwargs)

        assert config.synthesise_only is True
        assert config.audio_file == transcript

    def test_synthesise_only_no_transcribe_creds_needed(self, valid_kwargs, monkeypatch, tmp_path):
        """--synthesise-only does not require Azure transcription credentials."""
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("Some transcript content")

        monkeypatch.delenv("AZURE_TRANSCRIBE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TRANSCRIBE_URL", raising=False)
        monkeypatch.setenv("AZURE_TEXT_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TEXT_URL", "https://test.example.com")
        valid_kwargs["synthesise_only"] = True
        valid_kwargs["audio_file"] = str(transcript)

        config = validate_cli_config(**valid_kwargs)

        assert config.synthesise_only is True

    def test_valid_config_returns_dataclass(self, valid_kwargs, monkeypatch):
        """Valid config returns ValidatedConfig dataclass."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")

        config = validate_cli_config(**valid_kwargs)

        assert config.parallel_workers == 15
        assert config.synthesise is False
        assert config.glossary is None
        assert config.transcribe_api_key == "test-key"


class TestValidationHappensBeforeWork:
    """Tests that parameter validation occurs BEFORE any FFmpeg/conversion work."""

    def test_invalid_param_fails_before_conversion(self, short_audio_file):
        """Invalid parameters should fail BEFORE any conversion/FFmpeg work starts."""
        import subprocess
        import sys

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
            timeout=10,
        )

        assert result.returncode != 0
        assert "cannot exceed 100" in result.stderr
        assert "Converting" not in result.stderr
        assert "extracting audio" not in result.stderr.lower()

    def test_missing_glossary_fails_before_conversion(self, short_audio_file):
        """Missing glossary file should fail BEFORE any conversion work."""
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
        assert "Converting" not in result.stderr

    def test_missing_env_vars_fails_before_conversion(self, short_audio_file):
        """Missing environment variables should fail BEFORE any conversion work."""
        import subprocess
        import sys

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
        assert "Converting" not in result.stderr


class TestMissingCredentials:
    """Tests for missing Azure credentials via settings .from_env()."""

    def test_missing_transcription_api_key(self, clean_env):
        """Raises ConfigurationError when AZURE_TRANSCRIBE_API_KEY is missing."""
        from transcriber import AzureTranscriptionSettings

        with pytest.raises(ConfigurationError):
            AzureTranscriptionSettings.from_env()

    def test_missing_transcription_api_url(self, clean_env, monkeypatch):
        """Raises ConfigurationError when AZURE_TRANSCRIBE_URL is missing."""
        from transcriber import AzureTranscriptionSettings

        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")

        with pytest.raises(ConfigurationError):
            AzureTranscriptionSettings.from_env()

    def test_missing_text_api_key(self, clean_env):
        """Raises ConfigurationError when AZURE_TEXT_API_KEY is missing."""
        from transcriber import AzureLLMSettings

        with pytest.raises(ConfigurationError):
            AzureLLMSettings.from_env()

    def test_missing_text_api_url(self, clean_env, monkeypatch):
        """Raises ConfigurationError when AZURE_TEXT_URL is missing."""
        from transcriber import AzureLLMSettings

        monkeypatch.setenv("AZURE_TEXT_API_KEY", "test-text-key")

        with pytest.raises(ConfigurationError):
            AzureLLMSettings.from_env()


class TestFileValidation:
    """Tests for file validation via the public API."""

    def test_file_not_found(self):
        """Raises AudioFileError for nonexistent audio file."""
        from unittest.mock import AsyncMock

        from transcriber import AudioFileError, transcribe_file

        mock_backend = AsyncMock()
        with pytest.raises(AudioFileError):
            transcribe_file("/nonexistent/path/to/audio.mp3", transcription_backend=mock_backend)


class TestInvalidApiCredentials:
    """Tests for invalid API credentials (requires network)."""

    async def test_invalid_api_key_returns_error(self, short_audio_file):
        """Invalid API key results in a TranscriptionError."""
        from transcriber import TranscriptionError
        from transcriber.backends import create_azure_transcription_backend

        backend = create_azure_transcription_backend(
            api_key="invalid-api-key-12345",
            api_url="https://invalid-endpoint.openai.azure.com/openai/deployments/test/audio/transcriptions?api-version=2025-03-01-preview",
        )
        try:
            with pytest.raises((TranscriptionError, Exception)):
                await backend.transcribe(short_audio_file)
        finally:
            await backend.aclose()

    async def test_invalid_url_returns_error(self, short_audio_file):
        """Invalid API URL results in a TranscriptionError."""
        from transcriber import TranscriptionError
        from transcriber.backends import create_azure_transcription_backend

        backend = create_azure_transcription_backend(
            api_key="test-key",
            api_url="https://this-endpoint-does-not-exist-12345.openai.azure.com/test",
        )
        try:
            with pytest.raises((TranscriptionError, Exception)):
                await backend.transcribe(short_audio_file)
        finally:
            await backend.aclose()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_audio_file(self):
        """Empty audio file is handled gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name

        try:
            from transcriber._audio import get_audio_duration

            duration = get_audio_duration(temp_path)
            assert duration is None or duration == 0
        finally:
            os.unlink(temp_path)

    def test_corrupted_audio_file(self):
        """Corrupted audio file is handled gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"This is not valid MP3 data, just random bytes: " + os.urandom(1000))
            temp_path = f.name

        try:
            from transcriber._audio import get_audio_duration

            duration = get_audio_duration(temp_path)
            assert duration is None
        finally:
            os.unlink(temp_path)


class TestWhisperImport:
    """Tests for Whisper lazy import."""

    def test_import_whisper_missing_raises(self, monkeypatch):
        """_import_whisper raises ConfigurationError when whisper not installed."""
        from transcriber.backends._whisper import _import_whisper

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "whisper":
                raise ImportError("No module named 'whisper'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ConfigurationError) as exc_info:
            _import_whisper()

        errors_text = " ".join(exc_info.value.errors)
        assert "openai-whisper" in errors_text


class TestLocalTranscription:
    """Tests for local Whisper transcription functions."""

    def test_format_whisper_output_basic(self):
        """format_whisper_output formats segments with timestamps."""
        segments = [
            {"start": 0.0, "end": 5.23, "text": "Hello, welcome to the meeting."},
            {"start": 5.23, "end": 10.50, "text": "Thank you for having me."},
        ]
        result = format_whisper_output(segments)
        assert "[0.00s - 5.23s] Hello, welcome to the meeting." in result
        assert "[5.23s - 10.50s] Thank you for having me." in result

    def test_format_whisper_output_with_time_offset(self):
        """format_whisper_output adds time offset to timestamps."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "Some text"},
        ]
        result = format_whisper_output(segments, time_offset=900)
        assert "[900.00s - 905.00s] Some text" in result

    def test_format_whisper_output_empty_segments(self):
        """format_whisper_output handles empty segment list."""
        result = format_whisper_output([])
        assert result == ""

    def test_format_whisper_output_strips_whitespace(self):
        """format_whisper_output strips whitespace from segment text."""
        segments = [
            {"start": 0.0, "end": 5.0, "text": "  Hello world  "},
        ]
        result = format_whisper_output(segments)
        assert "[0.00s - 5.00s] Hello world" in result

    def test_format_whisper_output_skips_empty_text(self):
        """format_whisper_output skips segments with empty text."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": ""},
            {"start": 1.0, "end": 2.0, "text": "Real content"},
            {"start": 2.0, "end": 3.0, "text": "   "},
        ]
        result = format_whisper_output(segments)
        lines = result.strip().split("\n")
        assert len(lines) == 1
        assert "Real content" in result


class TestLocalModeValidation:
    """Tests for local mode validation in validate_cli_config."""

    @pytest.fixture
    def valid_audio_file(self, short_audio_file):
        """Return a valid audio file path."""
        return short_audio_file

    @pytest.fixture
    def local_kwargs(self, valid_audio_file):
        """Create keyword args for local mode."""
        return {
            "audio_file": valid_audio_file,
            "output_file": None,
            "glossary": None,
            "synthesise": False,
            "synthesise_only": False,
            "parallel_workers": 15,
            "local": True,
            "model": "base",
        }

    def test_local_mode_no_azure_credentials_required(self, local_kwargs, monkeypatch):
        """--local works without AZURE_TRANSCRIBE_* env vars."""
        monkeypatch.delenv("AZURE_TRANSCRIBE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TRANSCRIBE_URL", raising=False)
        monkeypatch.delenv("AZURE_TEXT_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TEXT_URL", raising=False)

        # Mock whisper import
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "whisper":
                import types

                mock_whisper = types.ModuleType("whisper")
                return mock_whisper
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        config = validate_cli_config(**local_kwargs)
        assert config.local_mode is True
        assert config.transcribe_api_key == ""
        assert config.transcribe_url == ""

    def test_azure_mode_still_requires_credentials(self, local_kwargs, monkeypatch):
        """Default (non-local) mode still requires Azure credentials."""
        local_kwargs["local"] = False
        monkeypatch.delenv("AZURE_TRANSCRIBE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TRANSCRIBE_URL", raising=False)

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**local_kwargs)

        errors_text = " ".join(exc_info.value.errors)
        assert "AZURE_TRANSCRIBE_API_KEY" in errors_text

    def test_local_with_glossary_requires_text_api(self, local_kwargs, monkeypatch):
        """--local with --glossary still requires Azure text API credentials."""
        local_kwargs["glossary"] = "/some/glossary.txt"
        monkeypatch.delenv("AZURE_TRANSCRIBE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TRANSCRIBE_URL", raising=False)
        monkeypatch.delenv("AZURE_TEXT_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TEXT_URL", raising=False)

        # Mock whisper import
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "whisper":
                import types

                return types.ModuleType("whisper")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**local_kwargs)

        errors_text = " ".join(exc_info.value.errors)
        # Should require text API but NOT transcription API
        assert "AZURE_TEXT_API_KEY" in errors_text
        assert "AZURE_TRANSCRIBE_API_KEY" not in errors_text

    def test_local_with_synthesise_requires_text_api(self, local_kwargs, monkeypatch):
        """--local with --synthesise still requires Azure text API credentials."""
        local_kwargs["synthesise"] = True
        monkeypatch.delenv("AZURE_TRANSCRIBE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TRANSCRIBE_URL", raising=False)
        monkeypatch.delenv("AZURE_TEXT_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TEXT_URL", raising=False)

        # Mock whisper import
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "whisper":
                import types

                return types.ModuleType("whisper")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**local_kwargs)

        errors_text = " ".join(exc_info.value.errors)
        assert "AZURE_TEXT_API_KEY" in errors_text

    def test_local_mode_whisper_not_installed_error(self, local_kwargs, monkeypatch):
        """--local mode fails with helpful message when whisper not installed."""
        monkeypatch.delenv("AZURE_TRANSCRIBE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TRANSCRIBE_URL", raising=False)

        # Mock whisper import to fail
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "whisper":
                raise ImportError("No module named 'whisper'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**local_kwargs)

        errors_text = " ".join(exc_info.value.errors)
        assert "openai-whisper" in errors_text


class TestTextFileAutoDetection:
    """Tests for auto-detecting text files and routing to synthesis-only mode."""

    @pytest.fixture
    def valid_audio_file(self, short_audio_file):
        """Return a valid audio file path."""
        return short_audio_file

    @pytest.fixture
    def valid_kwargs(self, valid_audio_file):
        """Create valid keyword args for validate_cli_config."""
        return {
            "audio_file": valid_audio_file,
            "output_file": None,
            "glossary": None,
            "synthesise": False,
            "synthesise_only": False,
            "parallel_workers": 15,
            "local": False,
            "model": "base",
        }

    @pytest.mark.parametrize("ext", [".txt", ".md", ".srt", ".vtt"])
    def test_text_file_auto_enables_synthesise_only(self, valid_kwargs, monkeypatch, tmp_path, ext):
        """Text file extensions auto-enable synthesis-only mode."""
        transcript = tmp_path / f"meeting{ext}"
        transcript.write_text("Some transcript content")

        monkeypatch.setenv("AZURE_TEXT_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TEXT_URL", "https://test.example.com")
        valid_kwargs["audio_file"] = str(transcript)

        config = validate_cli_config(**valid_kwargs)

        assert config.synthesise_only is True

    def test_text_file_with_synthesise_flag_coerced(self, valid_kwargs, monkeypatch, tmp_path):
        """--synthesise with a text file is coerced to synthesis-only (no error)."""
        transcript = tmp_path / "notes.txt"
        transcript.write_text("Some notes")

        monkeypatch.setenv("AZURE_TEXT_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TEXT_URL", "https://test.example.com")
        valid_kwargs["audio_file"] = str(transcript)
        valid_kwargs["synthesise"] = True

        config = validate_cli_config(**valid_kwargs)

        assert config.synthesise_only is True

    def test_text_file_no_transcription_creds_needed(self, valid_kwargs, monkeypatch, tmp_path):
        """Text files do not require Azure transcription credentials."""
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("Some transcript content")

        monkeypatch.delenv("AZURE_TRANSCRIBE_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TRANSCRIBE_URL", raising=False)
        monkeypatch.setenv("AZURE_TEXT_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TEXT_URL", "https://test.example.com")
        valid_kwargs["audio_file"] = str(transcript)

        config = validate_cli_config(**valid_kwargs)

        assert config.synthesise_only is True

    def test_text_file_requires_llm_creds(self, valid_kwargs, monkeypatch, tmp_path):
        """Text files require LLM credentials for synthesis."""
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("Some transcript content")

        monkeypatch.delenv("AZURE_TEXT_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_TEXT_URL", raising=False)
        valid_kwargs["audio_file"] = str(transcript)

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**valid_kwargs)

        errors_text = " ".join(exc_info.value.errors)
        assert "AZURE_TEXT_API_KEY" in errors_text

    def test_text_file_not_found(self, valid_kwargs, monkeypatch):
        """Non-existent text file raises ConfigurationError."""
        monkeypatch.setenv("AZURE_TEXT_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TEXT_URL", "https://test.example.com")
        valid_kwargs["audio_file"] = "/nonexistent/transcript.txt"

        with pytest.raises(ConfigurationError) as exc_info:
            validate_cli_config(**valid_kwargs)

        assert any("Transcript file not found" in e for e in exc_info.value.errors)

    def test_audio_extension_not_auto_detected(self, valid_kwargs, monkeypatch):
        """Audio file extensions are NOT auto-detected as text files."""
        monkeypatch.setenv("AZURE_TRANSCRIBE_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_TRANSCRIBE_URL", "https://test.example.com")

        config = validate_cli_config(**valid_kwargs)

        assert config.synthesise_only is False
