"""
Pytest configuration and fixtures for jbb_transcriber tests.
"""

import os
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables from tests/.env
_tests_dir = Path(__file__).parent
_env_path = _tests_dir / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)


@pytest.fixture(scope="session")
def fixtures_dir():
    """Path to the test fixtures directory."""
    return _tests_dir / "fixtures"


@pytest.fixture(scope="session")
def litellm_config():
    """Shared LiteLLM proxy configuration.

    Resolves JBB_TRANSCRIBER_API_KEY → OPENAI_API_KEY and
    JBB_TRANSCRIBER_BASE_URL → OPENAI_BASE_URL, matching the app's own logic.
    Skips tests if credentials are not configured.
    """
    api_key = os.getenv("JBB_TRANSCRIBER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("JBB_TRANSCRIBER_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("JBB_TRANSCRIBER_MODEL")

    if not api_key or not base_url:
        pytest.skip("LiteLLM proxy credentials not configured in tests/.env")
    if not model:
        pytest.skip(
            "JBB_TRANSCRIBER_MODEL not configured in tests/.env — "
            'add: JBB_TRANSCRIBER_MODEL="your-model-name"'
        )

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


@pytest.fixture
def azure_transcription_backend(litellm_config):
    """An AzureTranscriptionBackend built from test credentials."""
    from jbb_transcriber.backends import create_azure_transcription_backend

    return create_azure_transcription_backend(
        api_key=litellm_config["api_key"],
        api_url=litellm_config["base_url"],
        model=litellm_config["model"],
    )


@pytest.fixture(scope="session")
def azure_text_config(litellm_config):
    """LiteLLM config extended with text model for glossary/synthesis tests.

    Skips if JBB_TRANSCRIBER_TEXT_MODEL is not configured.
    """
    text_model = os.getenv("JBB_TRANSCRIBER_TEXT_MODEL")
    if not text_model:
        pytest.skip(
            "JBB_TRANSCRIBER_TEXT_MODEL not configured in tests/.env — "
            'add: JBB_TRANSCRIBER_TEXT_MODEL="your-text-model-name"'
        )

    return {
        **litellm_config,
        "text_model": text_model,
    }


@pytest.fixture
def azure_llm_backend(azure_text_config):
    """An AzureLLMBackend built from test credentials."""
    from jbb_transcriber.backends import create_azure_llm_backend

    return create_azure_llm_backend(
        api_key=azure_text_config["api_key"],
        api_url=azure_text_config["base_url"],
        model=azure_text_config["text_model"],
    )


@pytest.fixture(scope="session")
def short_audio_file(fixtures_dir):
    """Path to a short audio sample (<30 sec) with clear speech.

    Skips tests if the audio file is not present.
    """
    audio_path = fixtures_dir / "short_speech.mp3"
    if not audio_path.exists():
        pytest.skip(f"Test audio file not found: {audio_path}")
    return str(audio_path)


@pytest.fixture(scope="session")
def multi_speaker_audio(fixtures_dir):
    """Path to audio with multiple speakers for diarization tests.

    Skips tests if the audio file is not present.
    """
    audio_path = fixtures_dir / "two_speakers.mp3"
    if not audio_path.exists():
        pytest.skip(f"Test audio file not found: {audio_path}")
    return str(audio_path)


@pytest.fixture(scope="session")
def long_audio_file(fixtures_dir):
    """Path to audio >23 min 20 sec for chunking tests.

    Skips tests if the audio file is not present.
    """
    audio_path = fixtures_dir / "long_recording.mp3"
    if not audio_path.exists():
        pytest.skip(f"Test audio file not found: {audio_path}")
    return str(audio_path)


@pytest.fixture
def sample_glossary():
    """Creates a temporary glossary file with sample terms."""
    content = """# Sample Glossary

Technical Terms:
- PyAV: Python bindings for FFmpeg
- Azure OpenAI: Microsoft's cloud AI service
- Diarization: Speaker identification in audio

Names:
- Claude: Anthropic's AI assistant

Acronyms:
- API: Application Programming Interface
- CLI: Command Line Interface
- LLM: Large Language Model
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_output_file():
    """Provides a temporary file path for output, cleans up after test."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        temp_path = f.name

    # Remove the file so tests can create it
    os.unlink(temp_path)

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all jbb_transcriber and OpenAI env vars for isolation in negative tests."""
    monkeypatch.delenv("JBB_TRANSCRIBER_API_KEY", raising=False)
    monkeypatch.delenv("JBB_TRANSCRIBER_BASE_URL", raising=False)
    monkeypatch.delenv("JBB_TRANSCRIBER_MODEL", raising=False)
    monkeypatch.delenv("JBB_TRANSCRIBER_TEXT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    # Legacy vars (in case any are still set in the shell)
    monkeypatch.delenv("AZURE_TRANSCRIBE_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_TRANSCRIBE_URL", raising=False)
    monkeypatch.delenv("AZURE_TRANSCRIBE_MODEL", raising=False)
    monkeypatch.delenv("AZURE_TEXT_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_TEXT_URL", raising=False)
    monkeypatch.delenv("AZURE_TEXT_MODEL", raising=False)
