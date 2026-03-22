"""Tests for retry logic, error classification, and backoff behaviour."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from transcriber._exceptions import LLMError, TranscriptionError
from transcriber._retry import (
    _extract_retry_after,
    is_transient_http_error,
    retry_with_backoff,
)

# ---------------------------------------------------------------------------
# is_transient_http_error classification tests
# ---------------------------------------------------------------------------


class TestIsTransientHttpError:
    """Tests for is_transient_http_error predicate."""

    def test_timeout_is_transient(self) -> None:
        assert is_transient_http_error(httpx.TimeoutException("timeout")) is True

    def test_connection_error_is_transient(self) -> None:
        assert is_transient_http_error(httpx.ConnectError("connect failed")) is True

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_retryable_http_status(self, status: int) -> None:
        request = httpx.Request("POST", "https://example.com")
        response = httpx.Response(status, request=request)
        exc = httpx.HTTPStatusError("error", request=request, response=response)
        assert is_transient_http_error(exc) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 422])
    def test_non_retryable_http_status(self, status: int) -> None:
        request = httpx.Request("POST", "https://example.com")
        response = httpx.Response(status, request=request)
        exc = httpx.HTTPStatusError("error", request=request, response=response)
        assert is_transient_http_error(exc) is False

    def test_generic_http_error_is_transient(self) -> None:
        exc = httpx.HTTPError("something went wrong")
        assert is_transient_http_error(exc) is True

    def test_transcription_error_retryable(self) -> None:
        exc = TranscriptionError("fail", status_code=503)
        assert is_transient_http_error(exc) is True

    def test_transcription_error_not_retryable(self) -> None:
        exc = TranscriptionError("fail", status_code=401)
        assert is_transient_http_error(exc) is False

    def test_transcription_error_no_status(self) -> None:
        exc = TranscriptionError("connection dropped")
        assert is_transient_http_error(exc) is True

    def test_llm_error_retryable(self) -> None:
        exc = LLMError("fail", status_code=429)
        assert is_transient_http_error(exc) is True

    def test_llm_error_not_retryable(self) -> None:
        exc = LLMError("fail", status_code=404)
        assert is_transient_http_error(exc) is False

    def test_unknown_exception_not_retried(self) -> None:
        assert is_transient_http_error(ValueError("oops")) is False


# ---------------------------------------------------------------------------
# _extract_retry_after tests
# ---------------------------------------------------------------------------


class TestExtractRetryAfter:
    """Tests for _extract_retry_after helper."""

    def test_no_response(self) -> None:
        assert _extract_retry_after(ValueError("x")) is None

    def test_no_header(self) -> None:
        request = httpx.Request("POST", "https://example.com")
        response = httpx.Response(429, request=request)
        exc = httpx.HTTPStatusError("error", request=request, response=response)
        assert _extract_retry_after(exc) is None

    def test_integer_seconds(self) -> None:
        request = httpx.Request("POST", "https://example.com")
        response = httpx.Response(429, request=request, headers={"Retry-After": "30"})
        exc = httpx.HTTPStatusError("error", request=request, response=response)
        assert _extract_retry_after(exc) == 30.0

    def test_float_seconds(self) -> None:
        request = httpx.Request("POST", "https://example.com")
        response = httpx.Response(429, request=request, headers={"Retry-After": "2.5"})
        exc = httpx.HTTPStatusError("error", request=request, response=response)
        assert _extract_retry_after(exc) == 2.5

    def test_unparseable_value(self) -> None:
        request = httpx.Request("POST", "https://example.com")
        response = httpx.Response(
            429, request=request, headers={"Retry-After": "Thu, 01 Jan 2099 00:00:00 GMT"}
        )
        exc = httpx.HTTPStatusError("error", request=request, response=response)
        assert _extract_retry_after(exc) is None


# ---------------------------------------------------------------------------
# retry_with_backoff tests (async)
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    """Tests for the async retry_with_backoff function."""

    async def test_success_on_first_try(self) -> None:
        fn = AsyncMock(return_value="ok")
        result = await retry_with_backoff(fn, max_retries=3, operation_name="test")
        assert result == "ok"
        assert fn.call_count == 1

    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_success_after_transient_failure(self, mock_sleep: AsyncMock) -> None:
        fn = AsyncMock(side_effect=[ValueError("transient"), "ok"])
        result = await retry_with_backoff(
            fn,
            max_retries=3,
            exceptions=(ValueError,),
            operation_name="test",
        )
        assert result == "ok"
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_all_retries_exhausted(self, mock_sleep: AsyncMock) -> None:
        exc = ValueError("persistent")
        fn = AsyncMock(side_effect=exc)

        with pytest.raises(ValueError, match="persistent"):
            await retry_with_backoff(
                fn,
                max_retries=3,
                exceptions=(ValueError,),
                operation_name="test",
            )

        assert fn.call_count == 3
        assert mock_sleep.call_count == 2  # backoff before attempts 2 and 3

    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_should_retry_false_stops_immediately(self, mock_sleep: AsyncMock) -> None:
        """Permanent errors are not retried when should_retry returns False."""
        exc = ValueError("permanent")
        fn = AsyncMock(side_effect=exc)

        with pytest.raises(ValueError, match="permanent"):
            await retry_with_backoff(
                fn,
                max_retries=5,
                exceptions=(ValueError,),
                operation_name="test",
                should_retry=lambda _e: False,
            )

        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_should_retry_true_allows_retries(self, mock_sleep: AsyncMock) -> None:
        fn = AsyncMock(side_effect=[ValueError("t1"), ValueError("t2"), "ok"])
        result = await retry_with_backoff(
            fn,
            max_retries=3,
            exceptions=(ValueError,),
            operation_name="test",
            should_retry=lambda _e: True,
        )
        assert result == "ok"
        assert fn.call_count == 3

    @patch("transcriber._retry.random.uniform", return_value=1.0)
    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_jitter_disabled_uses_exact_backoff(
        self, mock_sleep: AsyncMock, mock_uniform: MagicMock
    ) -> None:
        fn = AsyncMock(side_effect=[ValueError("x"), "ok"])
        await retry_with_backoff(
            fn,
            max_retries=2,
            base_delay=2.0,
            exceptions=(ValueError,),
            jitter=False,
        )
        # With jitter=False, random.uniform should not be called
        mock_uniform.assert_not_called()
        # Sleep should be called with exactly base_delay * 2^0 = 2.0
        mock_sleep.assert_called_once_with(2.0)

    @patch("transcriber._retry.random.uniform", return_value=1.5)
    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_jitter_enabled_varies_delay(
        self, mock_sleep: AsyncMock, mock_uniform: MagicMock
    ) -> None:
        fn = AsyncMock(side_effect=[ValueError("x"), "ok"])
        await retry_with_backoff(
            fn,
            max_retries=2,
            base_delay=2.0,
            exceptions=(ValueError,),
            jitter=True,
        )
        mock_uniform.assert_called_once_with(0.5, 1.5)
        # base_delay * 2^0 * 1.5 = 3.0
        mock_sleep.assert_called_once_with(3.0)

    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_after_header_respected(self, mock_sleep: AsyncMock) -> None:
        request = httpx.Request("POST", "https://example.com")
        response = httpx.Response(429, request=request, headers={"Retry-After": "10"})
        exc = httpx.HTTPStatusError("error", request=request, response=response)

        fn = AsyncMock(side_effect=[exc, "ok"])
        result = await retry_with_backoff(
            fn,
            max_retries=2,
            base_delay=2.0,
            exceptions=(httpx.HTTPError,),
            jitter=False,
        )
        assert result == "ok"
        # max(base_delay=2.0, retry_after=10.0) = 10.0
        mock_sleep.assert_called_once_with(10.0)

    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_final_failure_logged_at_error(
        self, mock_sleep: AsyncMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        fn = AsyncMock(side_effect=ValueError("boom"))
        with caplog.at_level("ERROR", logger="transcriber._retry"):
            with pytest.raises(ValueError, match="boom"):
                await retry_with_backoff(
                    fn,
                    max_retries=2,
                    exceptions=(ValueError,),
                    operation_name="my_op",
                )

        assert any("my_op failed after 2 attempts" in r.message for r in caplog.records)

    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_non_retryable_error_logged_at_error(
        self, mock_sleep: AsyncMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        fn = AsyncMock(side_effect=ValueError("auth"))
        with caplog.at_level("ERROR", logger="transcriber._retry"):
            with pytest.raises(ValueError, match="auth"):
                await retry_with_backoff(
                    fn,
                    max_retries=3,
                    exceptions=(ValueError,),
                    operation_name="my_op",
                    should_retry=lambda _e: False,
                )

        assert any("non-retryable error" in r.message for r in caplog.records)

    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_http_status_logged_in_warning(
        self, mock_sleep: AsyncMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """HTTP status code is included in the per-attempt warning log."""
        request = httpx.Request("POST", "https://example.com")
        response = httpx.Response(503, request=request)
        exc = httpx.HTTPStatusError("error", request=request, response=response)

        fn = AsyncMock(side_effect=[exc, "ok"])
        with caplog.at_level("WARNING", logger="transcriber._retry"):
            await retry_with_backoff(
                fn,
                max_retries=2,
                exceptions=(httpx.HTTPError,),
                operation_name="svc",
            )

        assert any("HTTP 503" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Integration-style tests with mock HTTP
# ---------------------------------------------------------------------------


class TestRetryIntegration:
    """Integration tests: backend raises proper exceptions, retry_with_backoff retries them."""

    def _make_openai_status_error(self, status_code: int, text: str) -> openai.APIStatusError:
        """Build an openai.APIStatusError for the given status code."""
        import openai

        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.text = text
        mock_response.headers = {}
        mock_response.request = MagicMock()
        exc_classes = {
            401: openai.AuthenticationError,
            400: openai.BadRequestError,
            429: openai.RateLimitError,
            503: openai.InternalServerError,
        }
        cls = exc_classes.get(status_code, openai.APIStatusError)
        return cls(str(status_code), response=mock_response, body=None)

    def _make_mock_transcription_client(self, side_effects: list) -> MagicMock:
        """A mock openai client whose audio.transcriptions.create yields side_effects."""
        client = MagicMock()
        client.audio = MagicMock()
        client.audio.transcriptions = MagicMock()
        client.audio.transcriptions.create = AsyncMock(side_effect=side_effects)
        return client

    def _make_mock_llm_client(self, side_effects: list) -> MagicMock:
        """A mock openai client whose chat.completions.create yields side_effects."""
        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=side_effects)
        return client

    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_transcription_retries_on_503(self, mock_sleep: AsyncMock) -> None:
        """503 errors from backend are retried via retry_with_backoff."""
        import tempfile

        from transcriber.backends import create_azure_transcription_backend

        exc_503 = self._make_openai_status_error(503, "Service Unavailable")

        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"text": "Hello world"}

        mock_client = self._make_mock_transcription_client([exc_503, exc_503, mock_result])

        backend = create_azure_transcription_backend(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        backend._client = mock_client
        backend._owns_client = False

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            f.write(b"fake audio")
            f.flush()

            result = await retry_with_backoff(
                lambda: backend.transcribe(f.name),
                max_retries=3,
                exceptions=(TranscriptionError, httpx.HTTPError),
                operation_name="transcription",
                should_retry=is_transient_http_error,
            )

        assert result == "Hello world"
        assert mock_client.audio.transcriptions.create.await_count == 3

    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_transcription_no_retry_on_401(self, mock_sleep: AsyncMock) -> None:
        """401 errors from backend are not retried (permanent)."""
        import tempfile

        from transcriber.backends import create_azure_transcription_backend

        exc_401 = self._make_openai_status_error(401, "Unauthorized")

        mock_client = self._make_mock_transcription_client([exc_401])

        backend = create_azure_transcription_backend(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        backend._client = mock_client
        backend._owns_client = False

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            f.write(b"fake audio")
            f.flush()
            with pytest.raises(TranscriptionError) as exc_info:
                await backend.transcribe(f.name)

        assert exc_info.value.status_code == 401
        assert mock_client.audio.transcriptions.create.await_count == 1

    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_llm_retries_on_429(self, mock_sleep: AsyncMock) -> None:
        """429 errors from LLM backend are retried."""
        from transcriber.backends import create_azure_llm_backend

        exc_429 = self._make_openai_status_error(429, "Too Many Requests")

        mock_message = MagicMock()
        mock_message.content = "corrected text"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_result = MagicMock()
        mock_result.choices = [mock_choice]

        mock_client = self._make_mock_llm_client([exc_429, mock_result])

        backend = create_azure_llm_backend(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        backend._client = mock_client
        backend._owns_client = False

        result = await retry_with_backoff(
            lambda: backend.complete("test prompt"),
            max_retries=3,
            exceptions=(LLMError, httpx.HTTPError),
            operation_name="llm",
            should_retry=is_transient_http_error,
        )

        assert result == "corrected text"
        assert mock_client.chat.completions.create.await_count == 2

    @patch("transcriber._retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_llm_no_retry_on_400(self, mock_sleep: AsyncMock) -> None:
        """400 errors from LLM backend are not retried."""
        from transcriber.backends import create_azure_llm_backend

        exc_400 = self._make_openai_status_error(400, "Bad Request")

        mock_client = self._make_mock_llm_client([exc_400])

        backend = create_azure_llm_backend(
            api_key="key", api_url="https://example.com", model="test-model"
        )
        backend._client = mock_client
        backend._owns_client = False

        with pytest.raises(LLMError) as exc_info:
            await backend.complete("test prompt")

        assert exc_info.value.status_code == 400
        assert mock_client.chat.completions.create.await_count == 1


# ---------------------------------------------------------------------------
# Exception classification tests
# ---------------------------------------------------------------------------


class TestExceptionClassification:
    """Tests for is_retryable on TranscriptionError and LLMError."""

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_transcription_error_retryable_statuses(self, status: int) -> None:
        assert TranscriptionError("x", status_code=status).is_retryable is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 422])
    def test_transcription_error_non_retryable_statuses(self, status: int) -> None:
        assert TranscriptionError("x", status_code=status).is_retryable is False

    def test_transcription_error_no_status_is_retryable(self) -> None:
        assert TranscriptionError("connection lost").is_retryable is True

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_llm_error_retryable_statuses(self, status: int) -> None:
        assert LLMError("x", status_code=status).is_retryable is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_llm_error_non_retryable_statuses(self, status: int) -> None:
        assert LLMError("x", status_code=status).is_retryable is False

    def test_llm_error_no_status_is_retryable(self) -> None:
        assert LLMError("timeout").is_retryable is True
