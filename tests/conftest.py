"""
Pytest configuration and fixtures for transcriber tests.
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
def azure_transcribe_config():
    """Azure transcription API configuration.

    Skips tests if credentials are not configured.
    """
    api_key = os.getenv("AZURE_TRANSCRIBE_API_KEY")
    api_url = os.getenv("AZURE_TRANSCRIBE_URL")

    if not api_key or not api_url:
        pytest.skip("Azure transcription credentials not configured in tests/.env")

    return {
        "transcribe_key": api_key,
        "transcribe_url": api_url,
    }


@pytest.fixture(scope="session")
def azure_text_config(azure_transcribe_config):
    """Azure text/chat API configuration for glossary correction.

    Includes transcription config plus text API credentials.
    Skips tests if text API credentials are not configured.
    """
    text_key = os.getenv("AZURE_TEXT_API_KEY")
    text_url = os.getenv("AZURE_TEXT_URL")

    if not text_key or not text_url:
        pytest.skip("Azure text API credentials not configured in tests/.env")

    return {
        **azure_transcribe_config,
        "text_key": text_key,
        "text_url": text_url,
    }


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
    """Remove Azure environment variables for negative tests."""
    monkeypatch.delenv("AZURE_TRANSCRIBE_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_TRANSCRIBE_URL", raising=False)
    monkeypatch.delenv("AZURE_TEXT_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_TEXT_URL", raising=False)
