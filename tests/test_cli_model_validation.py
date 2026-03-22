"""Unit tests for Story 1.2 — CLI model env-var validation.

FR8:  validate_cli_config reads TRANSCRIBER_MODEL
FR9:  validate_cli_config reads TRANSCRIBER_TEXT_MODEL
FR10: ConfigurationError raised with specific messages when model vars are missing
FR11: ValidatedConfig gains transcribe_model: str and text_model: str | None
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import transcriber
from transcriber._settings import AzureLLMSettings, AzureTranscriptionSettings
from transcriber.backends._azure import AzureLLMBackend, AzureTranscriptionBackend
from transcriber.cli import ValidatedConfig, _run_async, validate_cli_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_KEY = "sk-test-key"
_BASE_URL = "https://proxy.example.com/v1"
_TRANSCRIBE_MODEL = "gpt-4o-transcribe"
_TEXT_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_cloud_env(monkeypatch, *, transcribe_model: str | None = _TRANSCRIBE_MODEL) -> None:
    monkeypatch.setenv("TRANSCRIBER_API_KEY", _API_KEY)
    monkeypatch.setenv("TRANSCRIBER_BASE_URL", _BASE_URL)
    if transcribe_model is not None:
        monkeypatch.setenv("TRANSCRIBER_MODEL", transcribe_model)
    else:
        monkeypatch.delenv("TRANSCRIBER_MODEL", raising=False)


def _set_text_env(monkeypatch, *, text_model: str | None = _TEXT_MODEL) -> None:
    if text_model is not None:
        monkeypatch.setenv("TRANSCRIBER_TEXT_MODEL", text_model)
    else:
        monkeypatch.delenv("TRANSCRIBER_TEXT_MODEL", raising=False)


def _validate(
    *,
    audio_file: str,
    synthesise: bool = False,
    synthesise_only: bool = False,
    glossary: str | None = None,
    local: bool = False,
) -> ValidatedConfig:
    """Call validate_cli_config with audio-probe mocked out."""
    with (
        patch("transcriber._audio.probe_audio_stream", return_value=(True, None)),
        patch("transcriber._audio.log_audio_file_info"),
    ):
        return validate_cli_config(
            audio_file=audio_file,
            output_file=None,
            glossary=glossary,
            synthesise=synthesise,
            synthesise_only=synthesise_only,
            parallel_workers=5,
            local=local,
            model="base",
        )


# ---------------------------------------------------------------------------
# FR11: ValidatedConfig structure
# ---------------------------------------------------------------------------


class TestValidatedConfigFields:
    """ValidatedConfig gains transcribe_model and text_model fields (FR11)."""

    def test_has_transcribe_model_field(self):
        """`transcribe_model: str` field exists on ValidatedConfig."""
        field_names = {f.name for f in dataclasses.fields(ValidatedConfig)}
        assert "transcribe_model" in field_names

    def test_has_text_model_field(self):
        """`text_model: str | None` field exists on ValidatedConfig."""
        field_names = {f.name for f in dataclasses.fields(ValidatedConfig)}
        assert "text_model" in field_names


# ---------------------------------------------------------------------------
# FR8: TRANSCRIBER_MODEL validation in cloud transcription mode
# ---------------------------------------------------------------------------


class TestTranscribeModelValidation:
    """TRANSCRIBER_MODEL is required in cloud mode (FR8, FR10)."""

    def test_missing_transcribe_model_raises(self, tmp_path, monkeypatch):
        """Missing TRANSCRIBER_MODEL raises ConfigurationError."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio")
        _set_cloud_env(monkeypatch, transcribe_model=None)

        with pytest.raises(transcriber.ConfigurationError) as exc_info:
            _validate(audio_file=str(audio))

        assert any("TRANSCRIBER_MODEL" in e for e in exc_info.value.errors)

    def test_missing_transcribe_model_error_message_format(self, tmp_path, monkeypatch):
        """Error message uses the canonical hint format."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio")
        _set_cloud_env(monkeypatch, transcribe_model=None)

        with pytest.raises(transcriber.ConfigurationError) as exc_info:
            _validate(audio_file=str(audio))

        matching = [e for e in exc_info.value.errors if "TRANSCRIBER_MODEL" in e]
        assert matching, "Expected at least one error mentioning TRANSCRIBER_MODEL"
        assert "your-transcription-model-name" in matching[0]
        assert "~/.zshrc" in matching[0]

    def test_transcribe_model_stored_in_config(self, tmp_path, monkeypatch):
        """TRANSCRIBER_MODEL value is forwarded to ValidatedConfig.transcribe_model."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio")
        _set_cloud_env(monkeypatch, transcribe_model=_TRANSCRIBE_MODEL)

        result = _validate(audio_file=str(audio))

        assert result.transcribe_model == _TRANSCRIBE_MODEL

    def test_transcribe_model_not_required_in_synthesise_only_mode(self, tmp_path, monkeypatch):
        """TRANSCRIBER_MODEL is not required when --synthesise-only is used."""
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("Speaker: Hello world")
        monkeypatch.delenv("TRANSCRIBER_MODEL", raising=False)
        _set_cloud_env(monkeypatch, transcribe_model=None)  # removes TRANSCRIBER_MODEL
        monkeypatch.setenv("TRANSCRIBER_API_KEY", _API_KEY)
        monkeypatch.setenv("TRANSCRIBER_BASE_URL", _BASE_URL)
        _set_text_env(monkeypatch, text_model=_TEXT_MODEL)

        with patch("transcriber._audio.is_text_file", return_value=False):
            try:
                validate_cli_config(
                    audio_file=str(transcript),
                    output_file=None,
                    glossary=None,
                    synthesise=False,
                    synthesise_only=True,
                    parallel_workers=5,
                    local=False,
                    model="base",
                )
                model_error = False
            except transcriber.ConfigurationError as e:
                model_error = any("TRANSCRIBER_MODEL" in err for err in e.errors)

        assert not model_error, "TRANSCRIBER_MODEL should not be required in synthesise-only mode"


# ---------------------------------------------------------------------------
# FR9: TRANSCRIBER_TEXT_MODEL validation when text API is required
# ---------------------------------------------------------------------------


class TestTextModelValidation:
    """TRANSCRIBER_TEXT_MODEL required for --glossary/--synthesise/--synthesise-only (FR9, FR10)."""

    def _check_missing_text_model(
        self, tmp_path, monkeypatch, *, flag: str, feature_label: str
    ) -> None:
        """Common helper: assert error raised for missing TRANSCRIBER_TEXT_MODEL."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio")
        _set_cloud_env(monkeypatch, transcribe_model=_TRANSCRIBE_MODEL)
        _set_text_env(monkeypatch, text_model=None)

        kwargs: dict = {"audio_file": str(audio)}
        if flag == "--glossary":
            glossary = tmp_path / "glossary.txt"
            glossary.write_text("Term")
            kwargs["glossary"] = str(glossary)
        elif flag == "--synthesise":
            kwargs["synthesise"] = True
        elif flag == "--synthesise-only":
            audio.write_text("Speaker: Hello")  # make it a text file check pass
            kwargs["synthesise_only"] = True

        with pytest.raises(transcriber.ConfigurationError) as exc_info:
            _validate(**kwargs)

        errors = exc_info.value.errors
        matching = [e for e in errors if "TRANSCRIBER_TEXT_MODEL" in e]
        assert matching, f"Expected error about TRANSCRIBER_TEXT_MODEL for {flag}"
        assert feature_label in matching[0], (
            f"Expected feature label '{feature_label}' in error: {matching[0]}"
        )
        assert "your-text-model-name" in matching[0]

    def test_missing_text_model_with_glossary(self, tmp_path, monkeypatch):
        self._check_missing_text_model(
            tmp_path, monkeypatch, flag="--glossary", feature_label="--glossary"
        )

    def test_missing_text_model_with_synthesise(self, tmp_path, monkeypatch):
        self._check_missing_text_model(
            tmp_path, monkeypatch, flag="--synthesise", feature_label="--synthesise"
        )

    def test_missing_text_model_with_synthesise_only(self, tmp_path, monkeypatch):
        transcript = tmp_path / "t.txt"
        transcript.write_text("Hello")
        monkeypatch.setenv("TRANSCRIBER_API_KEY", _API_KEY)
        monkeypatch.setenv("TRANSCRIBER_BASE_URL", _BASE_URL)
        _set_text_env(monkeypatch, text_model=None)
        monkeypatch.delenv("TRANSCRIBER_MODEL", raising=False)

        with patch("transcriber._audio.is_text_file", return_value=False):
            with pytest.raises(transcriber.ConfigurationError) as exc_info:
                validate_cli_config(
                    audio_file=str(transcript),
                    output_file=None,
                    glossary=None,
                    synthesise=False,
                    synthesise_only=True,
                    parallel_workers=5,
                    local=False,
                    model="base",
                )

        errors = exc_info.value.errors
        matching = [e for e in errors if "TRANSCRIBER_TEXT_MODEL" in e]
        assert matching, "Expected error about TRANSCRIBER_TEXT_MODEL for --synthesise-only"
        assert "--synthesise-only" in matching[0]

    def test_text_model_stored_in_config(self, tmp_path, monkeypatch):
        """TRANSCRIBER_TEXT_MODEL value is forwarded to ValidatedConfig.text_model."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio")
        glossary = tmp_path / "glossary.txt"
        glossary.write_text("Term")
        _set_cloud_env(monkeypatch, transcribe_model=_TRANSCRIBE_MODEL)
        _set_text_env(monkeypatch, text_model=_TEXT_MODEL)

        result = _validate(audio_file=str(audio), glossary=str(glossary))

        assert result.text_model == _TEXT_MODEL

    def test_text_model_is_none_when_not_required(self, tmp_path, monkeypatch):
        """text_model is None when no text API feature is requested."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake audio")
        _set_cloud_env(monkeypatch, transcribe_model=_TRANSCRIBE_MODEL)
        monkeypatch.delenv("TRANSCRIBER_TEXT_MODEL", raising=False)

        result = _validate(audio_file=str(audio))

        assert result.text_model is None


# ---------------------------------------------------------------------------
# Epilog coverage: TRANSCRIBER_MODEL and TRANSCRIBER_TEXT_MODEL appear in --help
# ---------------------------------------------------------------------------


class TestEpilogContainsModelVars:
    """CLI epilog lists model env vars (AC: --help / epilog text)."""

    def test_epilog_contains_transcribe_model(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "transcriber", "--help"],
            capture_output=True,
            text=True,
        )
        assert "TRANSCRIBER_MODEL" in result.stdout

    def test_epilog_contains_text_model(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "transcriber", "--help"],
            capture_output=True,
            text=True,
        )
        assert "TRANSCRIBER_TEXT_MODEL" in result.stdout


# ---------------------------------------------------------------------------
# T6/T7: model values forwarded from ValidatedConfig into backend settings
# ---------------------------------------------------------------------------


class TestModelForwardingToBackend:
    """_run_async passes transcribe_model/text_model into backend settings (T6, T7)."""

    def _cloud_config(self, audio: Path, **overrides: object) -> ValidatedConfig:
        """Build a minimal cloud-mode ValidatedConfig."""
        defaults: dict[str, object] = {
            "audio_file": audio,
            "output_file": audio.with_suffix(".txt"),
            "glossary": None,
            "glossary_text": None,
            "synthesise": False,
            "synthesise_only": False,
            "parallel_workers": 1,
            "chunk_duration": 900,
            "provider": "azure",
            "local_mode": False,
            "whisper_model": "base",
            "api_key": "key",
            "base_url": "https://example.com",
            "transcribe_model": "default-transcribe-model",
            "text_model": None,
        }
        return ValidatedConfig(**{**defaults, **overrides})  # type: ignore[arg-type]

    async def test_transcribe_model_forwarded_to_settings(self, tmp_path: Path) -> None:
        """T6: _run_async passes transcribe_model to AzureTranscriptionSettings."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        config = self._cloud_config(audio, transcribe_model="my-exact-transcribe-model")

        with (
            patch(
                "transcriber.cli.AzureTranscriptionSettings",
                wraps=AzureTranscriptionSettings,
            ) as mock_t_cls,
            patch.object(AzureTranscriptionBackend, "aclose", new_callable=AsyncMock),
            patch("transcriber.transcribe", new_callable=AsyncMock),
        ):
            await _run_async(config)

        mock_t_cls.assert_called_once()
        assert mock_t_cls.call_args.kwargs["model"] == "my-exact-transcribe-model"

    async def test_text_model_forwarded_to_settings(self, tmp_path: Path) -> None:
        """T7: _run_async passes validated.text_model to AzureLLMSettings(model=)."""
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"x")
        config = self._cloud_config(
            audio,
            transcribe_model="transcribe-model",
            text_model="my-exact-text-model",
        )

        with (
            patch(
                "transcriber.cli.AzureLLMSettings",
                wraps=AzureLLMSettings,
            ) as mock_llm_cls,
            patch.object(AzureTranscriptionBackend, "aclose", new_callable=AsyncMock),
            patch.object(AzureLLMBackend, "aclose", new_callable=AsyncMock),
            patch("transcriber.transcribe", new_callable=AsyncMock),
        ):
            await _run_async(config)

        mock_llm_cls.assert_called_once()
        assert mock_llm_cls.call_args.kwargs["model"] == "my-exact-text-model"
