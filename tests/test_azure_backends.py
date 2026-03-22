"""Unit tests for Story 1.1: Azure backends migrated to AsyncOpenAI SDK.

Covers all acceptance criteria for the httpx → openai SDK migration:
- Settings require model with no default
- Factory functions require model param
- Backends create AsyncOpenAI (not httpx) clients
- transcribe() calls audio.transcriptions.create with correct params
- complete() calls chat.completions.create with typed response access
- No api-key header, ?api-version=, or /deployments/ in backend code
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from transcriber._settings import AzureLLMSettings, AzureTranscriptionSettings
from transcriber.backends._azure import (
    AzureLLMBackend,
    AzureTranscriptionBackend,
    create_azure_llm_backend,
    create_azure_transcription_backend,
)

# ---------------------------------------------------------------------------
# AC: Settings require model with no default
# ---------------------------------------------------------------------------


class TestSettingsModelRequired:
    """AzureTranscriptionSettings and AzureLLMSettings require model=."""

    def test_transcription_settings_requires_model(self) -> None:
        """AzureTranscriptionSettings raises TypeError when model is omitted."""
        with pytest.raises(TypeError):
            AzureTranscriptionSettings(api_key="key", api_url="https://example.com")  # type: ignore[call-arg]

    def test_llm_settings_requires_model(self) -> None:
        """AzureLLMSettings raises TypeError when model is omitted."""
        with pytest.raises(TypeError):
            AzureLLMSettings(api_key="key", api_url="https://example.com")  # type: ignore[call-arg]

    def test_transcription_settings_no_default_model(self) -> None:
        """AzureTranscriptionSettings.model has no default value."""
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(AzureTranscriptionSettings)}
        model_field = fields["model"]
        assert model_field.default is dataclasses.MISSING
        assert model_field.default_factory is dataclasses.MISSING  # type: ignore[misc]

    def test_llm_settings_no_default_model(self) -> None:
        """AzureLLMSettings.model has no default (old 'gpt-5.1' default removed)."""
        import dataclasses

        fields = {f.name: f for f in dataclasses.fields(AzureLLMSettings)}
        model_field = fields["model"]
        assert model_field.default is dataclasses.MISSING
        assert model_field.default_factory is dataclasses.MISSING  # type: ignore[misc]

    def test_transcription_settings_accepts_model(self) -> None:
        """AzureTranscriptionSettings is valid when model is provided."""
        s = AzureTranscriptionSettings(
            api_key="key", api_url="https://example.com", model="my-model"
        )
        assert s.model == "my-model"

    def test_llm_settings_accepts_model(self) -> None:
        """AzureLLMSettings is valid when model is provided."""
        s = AzureLLMSettings(api_key="key", api_url="https://example.com", model="my-model")
        assert s.model == "my-model"


# ---------------------------------------------------------------------------
# AC: Factory functions require model param
# ---------------------------------------------------------------------------


class TestFactoryModelRequired:
    """create_azure_*_backend() raises TypeError when model is omitted."""

    def test_transcription_factory_requires_model(self) -> None:
        """create_azure_transcription_backend raises TypeError without model."""
        with pytest.raises(TypeError):
            create_azure_transcription_backend(  # type: ignore[call-arg]
                api_key="key", api_url="https://example.com"
            )

    def test_llm_factory_requires_model(self) -> None:
        """create_azure_llm_backend raises TypeError without model."""
        with pytest.raises(TypeError):
            create_azure_llm_backend(  # type: ignore[call-arg]
                api_key="key", api_url="https://example.com"
            )

    def test_transcription_factory_passes_model(self) -> None:
        """create_azure_transcription_backend forwards model to settings."""
        backend = create_azure_transcription_backend(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        assert backend._settings.model == "test-model"

    def test_llm_factory_passes_model(self) -> None:
        """create_azure_llm_backend forwards model to settings."""
        backend = create_azure_llm_backend(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        assert backend._settings.model == "test-model"


# ---------------------------------------------------------------------------
# AC: Backends create AsyncOpenAI (not httpx) client
# ---------------------------------------------------------------------------


class TestBackendUsesAsyncOpenAI:
    """AzureTranscriptionBackend and AzureLLMBackend create AsyncOpenAI clients."""

    def test_transcription_backend_creates_async_openai(self) -> None:
        """AzureTranscriptionBackend.__init__ creates AsyncOpenAI, not httpx."""
        import openai

        settings = AzureTranscriptionSettings(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        backend = AzureTranscriptionBackend(settings)
        assert isinstance(backend._client, openai.AsyncOpenAI)

    def test_transcription_backend_client_has_correct_base_url(self) -> None:
        """AsyncOpenAI client uses settings.api_url as base_url."""
        settings = AzureTranscriptionSettings(
            api_key="key", api_url="https://proxy.example.com/v1", model="test-model"
        )
        backend = AzureTranscriptionBackend(settings)
        assert str(backend._client.base_url).startswith("https://proxy.example.com/v1")

    def test_transcription_backend_client_has_correct_api_key(self) -> None:
        """AsyncOpenAI client uses settings.api_key."""
        settings = AzureTranscriptionSettings(
            api_key="my-secret-key", api_url="https://example.com", model="test-model"
        )
        backend = AzureTranscriptionBackend(settings)
        assert backend._client.api_key == "my-secret-key"

    def test_llm_backend_creates_async_openai(self) -> None:
        """AzureLLMBackend.__init__ creates AsyncOpenAI, not httpx."""
        import openai

        settings = AzureLLMSettings(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        backend = AzureLLMBackend(settings)
        assert isinstance(backend._client, openai.AsyncOpenAI)

    def test_transcription_backend_accepts_injected_client(self) -> None:
        """AzureTranscriptionBackend accepts a pre-configured client for testing."""
        import openai

        settings = AzureTranscriptionSettings(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        mock_client = MagicMock(spec=openai.AsyncOpenAI)
        backend = AzureTranscriptionBackend(settings, client=mock_client)
        assert backend._client is mock_client
        assert not backend._owns_client

    def test_llm_backend_accepts_injected_client(self) -> None:
        """AzureLLMBackend accepts a pre-configured client for testing."""
        import openai

        settings = AzureLLMSettings(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        mock_client = MagicMock(spec=openai.AsyncOpenAI)
        backend = AzureLLMBackend(settings, client=mock_client)
        assert backend._client is mock_client
        assert not backend._owns_client


# ---------------------------------------------------------------------------
# AC: transcribe() calls audio.transcriptions.create with correct params
# ---------------------------------------------------------------------------


class TestTranscribeCallParams:
    """AzureTranscriptionBackend.transcribe() uses SDK correctly."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """An AsyncOpenAI mock with audio.transcriptions.create configured."""
        import openai

        client = MagicMock(spec=openai.AsyncOpenAI)
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {
            "segments": [{"speaker": "A", "text": "Hello.", "start": 0.0, "end": 1.0}],
            "text": "Hello.",
        }
        client.audio = MagicMock()
        client.audio.transcriptions = MagicMock()
        client.audio.transcriptions.create = AsyncMock(return_value=mock_result)
        return client

    @pytest.fixture
    def fake_audio(self, tmp_path: Path) -> str:
        """A throwaway audio file for unit tests — no fixtures dir required."""
        path = tmp_path / "fake_audio.mp3"
        path.write_bytes(b"fake-audio-content")
        return str(path)

    @pytest.fixture
    def backend(self, mock_client: MagicMock) -> AzureTranscriptionBackend:
        """AzureTranscriptionBackend with mocked openai client."""
        settings = AzureTranscriptionSettings(
            api_key="key", api_url="https://example.com", model="my-transcribe-model"
        )
        return AzureTranscriptionBackend(settings, client=mock_client)

    async def test_calls_audio_transcriptions_create(
        self, backend: AzureTranscriptionBackend, mock_client: MagicMock, fake_audio: str
    ) -> None:
        """transcribe() calls client.audio.transcriptions.create()."""
        await backend.transcribe(fake_audio)
        mock_client.audio.transcriptions.create.assert_awaited_once()

    async def test_passes_model_param(
        self, backend: AzureTranscriptionBackend, mock_client: MagicMock, fake_audio: str
    ) -> None:
        """transcribe() passes settings.model as model= to create()."""
        await backend.transcribe(fake_audio)
        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["model"] == "my-transcribe-model"

    async def test_passes_response_format_diarized_json(
        self, backend: AzureTranscriptionBackend, mock_client: MagicMock, fake_audio: str
    ) -> None:
        """transcribe() passes response_format='diarized_json'."""
        await backend.transcribe(fake_audio)
        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["response_format"] == "diarized_json"

    async def test_passes_chunking_strategy_auto(
        self, backend: AzureTranscriptionBackend, mock_client: MagicMock, fake_audio: str
    ) -> None:
        """transcribe() passes chunking_strategy='auto'."""
        await backend.transcribe(fake_audio)
        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["chunking_strategy"] == "auto"

    async def test_no_api_key_header(
        self, backend: AzureTranscriptionBackend, mock_client: MagicMock, fake_audio: str
    ) -> None:
        """transcribe() does not pass api-key in headers."""
        await backend.transcribe(fake_audio)
        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        # No explicit headers kwarg with api-key
        assert "headers" not in call_kwargs or "api-key" not in (call_kwargs.get("headers") or {})

    async def test_parses_diarized_segments(
        self, backend: AzureTranscriptionBackend, mock_client: MagicMock, fake_audio: str
    ) -> None:
        """transcribe() formats the diarized segments into text."""
        result = await backend.transcribe(fake_audio)
        assert "[0.00s - 1.00s] A: Hello." in result

    async def test_applies_time_offset(
        self, backend: AzureTranscriptionBackend, mock_client: MagicMock, fake_audio: str
    ) -> None:
        """transcribe() correctly applies time_offset to timestamps."""
        result = await backend.transcribe(fake_audio, time_offset=100)
        assert "[100.00s - 101.00s] A: Hello." in result


# ---------------------------------------------------------------------------
# AC: complete() calls chat.completions.create with typed response access
# ---------------------------------------------------------------------------


class TestCompleteCallParams:
    """AzureLLMBackend.complete() uses SDK correctly with typed response."""

    @pytest.fixture
    def mock_completion(self) -> MagicMock:
        """A typed ChatCompletion mock."""
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice

        message = MagicMock(spec=ChatCompletionMessage)
        message.content = "Corrected text output."
        choice = MagicMock(spec=Choice)
        choice.message = message
        completion = MagicMock(spec=ChatCompletion)
        completion.choices = [choice]
        return completion

    @pytest.fixture
    def mock_client(self, mock_completion: MagicMock) -> MagicMock:
        """An AsyncOpenAI mock with chat.completions.create configured."""
        import openai

        client = MagicMock(spec=openai.AsyncOpenAI)
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=mock_completion)
        return client

    @pytest.fixture
    def backend(self, mock_client: MagicMock) -> AzureLLMBackend:
        """AzureLLMBackend with mocked openai client."""
        settings = AzureLLMSettings(
            api_key="key", api_url="https://example.com", model="my-llm-model"
        )
        return AzureLLMBackend(settings, client=mock_client)

    async def test_calls_chat_completions_create(
        self, backend: AzureLLMBackend, mock_client: MagicMock
    ) -> None:
        """complete() calls client.chat.completions.create()."""
        await backend.complete("Hello")
        mock_client.chat.completions.create.assert_awaited_once()

    async def test_passes_model_param(
        self, backend: AzureLLMBackend, mock_client: MagicMock
    ) -> None:
        """complete() passes settings.model as model= to create()."""
        await backend.complete("Hello")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "my-llm-model"

    async def test_passes_messages_with_user_role(
        self, backend: AzureLLMBackend, mock_client: MagicMock
    ) -> None:
        """complete() passes messages=[{'role': 'user', 'content': prompt}]."""
        await backend.complete("My prompt text")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "My prompt text"}]

    async def test_passes_temperature(
        self, backend: AzureLLMBackend, mock_client: MagicMock
    ) -> None:
        """complete() forwards temperature to create()."""
        await backend.complete("Hello", temperature=0.7)
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7

    async def test_returns_typed_response_content(
        self, backend: AzureLLMBackend, mock_client: MagicMock
    ) -> None:
        """complete() returns result.choices[0].message.content (typed access)."""
        result = await backend.complete("Hello")
        assert result == "Corrected text output."

    async def test_parse_uses_typed_attributes(self, mock_completion: MagicMock) -> None:
        """_parse_completion_response accesses typed SDK object attributes, not dict."""
        result = AzureLLMBackend._parse_completion_response(mock_completion)
        assert result == "Corrected text output."
        # Verify access was via attribute, not dict-key
        _ = mock_completion.choices[0].message.content  # should not raise


# ---------------------------------------------------------------------------
# AC: No Azure-specific URL construction left in backends/_azure.py
# ---------------------------------------------------------------------------


class TestNoAzureSpecificCode:
    """Ensure no Azure-specific URL patterns remain in _azure.py."""

    def test_no_api_version_query_param(self) -> None:
        """backends/_azure.py contains no '?api-version=' string."""
        import inspect

        from transcriber.backends import _azure

        source = inspect.getsource(_azure)
        assert "?api-version=" not in source

    def test_no_deployments_path(self) -> None:
        """backends/_azure.py contains no '/deployments/' string."""
        import inspect

        from transcriber.backends import _azure

        source = inspect.getsource(_azure)
        assert "/deployments/" not in source

    def test_no_api_key_header_construction(self) -> None:
        """backends/_azure.py contains no 'api-key' header construction."""
        import inspect

        from transcriber.backends import _azure

        source = inspect.getsource(_azure)
        assert '"api-key"' not in source


# ---------------------------------------------------------------------------
# AC: Error handling — openai exceptions are wrapped correctly
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Openai SDK exceptions are converted to TranscriptionError/LLMError."""

    @pytest.fixture
    def fake_audio(self, tmp_path: Path) -> str:
        """A throwaway audio file for unit tests — no fixtures dir required."""
        path = tmp_path / "fake_audio.mp3"
        path.write_bytes(b"fake-audio-content")
        return str(path)

    @pytest.fixture
    def transcription_backend(self) -> AzureTranscriptionBackend:
        settings = AzureTranscriptionSettings(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        mock_client = MagicMock()
        mock_client.audio = MagicMock()
        mock_client.audio.transcriptions = MagicMock()
        return AzureTranscriptionBackend(settings, client=mock_client)

    @pytest.fixture
    def llm_backend(self) -> AzureLLMBackend:
        settings = AzureLLMSettings(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        return AzureLLMBackend(settings, client=mock_client)

    async def test_transcription_timeout_raises_transcription_error(
        self, transcription_backend: AzureTranscriptionBackend, fake_audio: str
    ) -> None:
        """APITimeoutError is wrapped in TranscriptionError."""
        import openai

        from transcriber._exceptions import TranscriptionError

        transcription_backend._client.audio.transcriptions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )
        with pytest.raises(TranscriptionError, match="timed out"):
            await transcription_backend.transcribe(fake_audio)

    async def test_transcription_status_error_raises_transcription_error(
        self, transcription_backend: AzureTranscriptionBackend, fake_audio: str
    ) -> None:
        """APIStatusError is wrapped in TranscriptionError with status_code."""
        import openai

        from transcriber._exceptions import TranscriptionError

        mock_response = MagicMock()
        mock_response.text = "Unauthorized"
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_response.request = MagicMock()

        transcription_backend._client.audio.transcriptions.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                "401 Unauthorized",
                response=mock_response,
                body=None,
            )
        )
        with pytest.raises(TranscriptionError) as exc_info:
            await transcription_backend.transcribe(fake_audio)
        assert exc_info.value.status_code == 401

    async def test_llm_timeout_raises_llm_error(self, llm_backend: AzureLLMBackend) -> None:
        """APITimeoutError in complete() is wrapped in LLMError."""
        import openai

        from transcriber._exceptions import LLMError

        llm_backend._client.chat.completions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )
        with pytest.raises(LLMError, match="timed out"):
            await llm_backend.complete("Hello")

    async def test_transcription_connection_error_raises_transcription_error(
        self, transcription_backend: AzureTranscriptionBackend, fake_audio: str
    ) -> None:
        """APIConnectionError is wrapped in TranscriptionError."""
        import openai

        from transcriber._exceptions import TranscriptionError

        transcription_backend._client.audio.transcriptions.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=MagicMock())
        )
        with pytest.raises(TranscriptionError, match="API request failed"):
            await transcription_backend.transcribe(fake_audio)


# ---------------------------------------------------------------------------
# M2: aclose() ownership contract
# ---------------------------------------------------------------------------


class TestAcloseLifecycle:
    """aclose() closes the owned client; skips the injected (not-owned) client."""

    async def test_transcription_aclose_closes_owned_client(self) -> None:
        """When _owns_client=True, aclose() awaits client.close()."""
        settings = AzureTranscriptionSettings(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        mock_client = AsyncMock()
        backend = AzureTranscriptionBackend(settings)
        backend._client = mock_client  # replace auto-created real client
        backend._owns_client = True
        await backend.aclose()
        mock_client.close.assert_awaited_once()

    async def test_transcription_aclose_skips_unowned_client(self) -> None:
        """When _owns_client=False, aclose() does NOT call client.close()."""
        settings = AzureTranscriptionSettings(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        mock_client = AsyncMock()
        backend = AzureTranscriptionBackend(settings, client=mock_client)
        assert not backend._owns_client
        await backend.aclose()
        mock_client.close.assert_not_awaited()

    async def test_llm_aclose_closes_owned_client(self) -> None:
        """When _owns_client=True, AzureLLMBackend.aclose() awaits client.close()."""
        settings = AzureLLMSettings(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        mock_client = AsyncMock()
        backend = AzureLLMBackend(settings)
        backend._client = mock_client
        backend._owns_client = True
        await backend.aclose()
        mock_client.close.assert_awaited_once()

    async def test_llm_aclose_skips_unowned_client(self) -> None:
        """When _owns_client=False, AzureLLMBackend.aclose() does NOT call client.close()."""
        settings = AzureLLMSettings(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        mock_client = AsyncMock()
        backend = AzureLLMBackend(settings, client=mock_client)
        assert not backend._owns_client
        await backend.aclose()
        mock_client.close.assert_not_awaited()
