"""Tests for retry logic, error classification, and backoff behaviour."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest
import requests
import requests.exceptions

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
        assert is_transient_http_error(requests.exceptions.Timeout()) is True

    def test_connection_error_is_transient(self) -> None:
        assert is_transient_http_error(requests.exceptions.ConnectionError()) is True

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_retryable_http_status(self, status: int) -> None:
        resp = MagicMock()
        resp.status_code = status
        exc = requests.exceptions.HTTPError(response=resp)
        assert is_transient_http_error(exc) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 422])
    def test_non_retryable_http_status(self, status: int) -> None:
        resp = MagicMock()
        resp.status_code = status
        exc = requests.exceptions.HTTPError(response=resp)
        assert is_transient_http_error(exc) is False

    def test_http_error_without_response(self) -> None:
        exc = requests.exceptions.HTTPError()
        assert is_transient_http_error(exc) is True

    def test_chunked_encoding_error_is_transient(self) -> None:
        assert is_transient_http_error(requests.exceptions.ChunkedEncodingError()) is True

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
        resp = MagicMock()
        resp.headers = {}
        exc = requests.exceptions.HTTPError(response=resp)
        assert _extract_retry_after(exc) is None

    def test_integer_seconds(self) -> None:
        resp = MagicMock()
        resp.headers = {"Retry-After": "30"}
        exc = requests.exceptions.HTTPError(response=resp)
        assert _extract_retry_after(exc) == 30.0

    def test_float_seconds(self) -> None:
        resp = MagicMock()
        resp.headers = {"Retry-After": "2.5"}
        exc = requests.exceptions.HTTPError(response=resp)
        assert _extract_retry_after(exc) == 2.5

    def test_unparseable_value(self) -> None:
        resp = MagicMock()
        resp.headers = {"Retry-After": "Thu, 01 Jan 2099 00:00:00 GMT"}
        exc = requests.exceptions.HTTPError(response=resp)
        assert _extract_retry_after(exc) is None


# ---------------------------------------------------------------------------
# retry_with_backoff tests
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    """Tests for the retry_with_backoff function."""

    def test_success_on_first_try(self) -> None:
        fn = MagicMock(return_value="ok")
        result = retry_with_backoff(fn, max_retries=3, operation_name="test")
        assert result == "ok"
        assert fn.call_count == 1

    @patch("transcriber._retry.time.sleep")
    def test_success_after_transient_failure(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[ValueError("transient"), "ok"])
        result = retry_with_backoff(
            fn,
            max_retries=3,
            exceptions=(ValueError,),
            operation_name="test",
        )
        assert result == "ok"
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    @patch("transcriber._retry.time.sleep")
    def test_all_retries_exhausted(self, mock_sleep: MagicMock) -> None:
        exc = ValueError("persistent")
        fn = MagicMock(side_effect=exc)

        with pytest.raises(ValueError, match="persistent"):
            retry_with_backoff(
                fn,
                max_retries=3,
                exceptions=(ValueError,),
                operation_name="test",
            )

        assert fn.call_count == 3
        assert mock_sleep.call_count == 2  # backoff before attempts 2 and 3

    @patch("transcriber._retry.time.sleep")
    def test_should_retry_false_stops_immediately(self, mock_sleep: MagicMock) -> None:
        """Permanent errors are not retried when should_retry returns False."""
        exc = ValueError("permanent")
        fn = MagicMock(side_effect=exc)

        with pytest.raises(ValueError, match="permanent"):
            retry_with_backoff(
                fn,
                max_retries=5,
                exceptions=(ValueError,),
                operation_name="test",
                should_retry=lambda _e: False,
            )

        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    @patch("transcriber._retry.time.sleep")
    def test_should_retry_true_allows_retries(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[ValueError("t1"), ValueError("t2"), "ok"])
        result = retry_with_backoff(
            fn,
            max_retries=3,
            exceptions=(ValueError,),
            operation_name="test",
            should_retry=lambda _e: True,
        )
        assert result == "ok"
        assert fn.call_count == 3

    @patch("transcriber._retry.random.uniform", return_value=1.0)
    @patch("transcriber._retry.time.sleep")
    def test_jitter_disabled_uses_exact_backoff(
        self, mock_sleep: MagicMock, mock_uniform: MagicMock
    ) -> None:
        fn = MagicMock(side_effect=[ValueError("x"), "ok"])
        retry_with_backoff(
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
    @patch("transcriber._retry.time.sleep")
    def test_jitter_enabled_varies_delay(
        self, mock_sleep: MagicMock, mock_uniform: MagicMock
    ) -> None:
        fn = MagicMock(side_effect=[ValueError("x"), "ok"])
        retry_with_backoff(
            fn,
            max_retries=2,
            base_delay=2.0,
            exceptions=(ValueError,),
            jitter=True,
        )
        mock_uniform.assert_called_once_with(0.5, 1.5)
        # base_delay * 2^0 * 1.5 = 3.0
        mock_sleep.assert_called_once_with(3.0)

    @patch("transcriber._retry.time.sleep")
    def test_retry_after_header_respected(self, mock_sleep: MagicMock) -> None:
        resp = MagicMock()
        resp.status_code = 429
        resp.headers = {"Retry-After": "10"}
        exc = requests.exceptions.HTTPError(response=resp)

        fn = MagicMock(side_effect=[exc, "ok"])
        result = retry_with_backoff(
            fn,
            max_retries=2,
            base_delay=2.0,
            exceptions=(requests.exceptions.RequestException,),
            jitter=False,
        )
        assert result == "ok"
        # max(base_delay=2.0, retry_after=10.0) = 10.0
        mock_sleep.assert_called_once_with(10.0)

    @patch("transcriber._retry.time.sleep")
    def test_final_failure_logged_at_error(
        self, mock_sleep: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        fn = MagicMock(side_effect=ValueError("boom"))
        with caplog.at_level("ERROR", logger="transcriber._retry"):
            with pytest.raises(ValueError, match="boom"):
                retry_with_backoff(
                    fn,
                    max_retries=2,
                    exceptions=(ValueError,),
                    operation_name="my_op",
                )

        assert any("my_op failed after 2 attempts" in r.message for r in caplog.records)

    @patch("transcriber._retry.time.sleep")
    def test_non_retryable_error_logged_at_error(
        self, mock_sleep: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        fn = MagicMock(side_effect=ValueError("auth"))
        with caplog.at_level("ERROR", logger="transcriber._retry"):
            with pytest.raises(ValueError, match="auth"):
                retry_with_backoff(
                    fn,
                    max_retries=3,
                    exceptions=(ValueError,),
                    operation_name="my_op",
                    should_retry=lambda _e: False,
                )

        assert any("non-retryable error" in r.message for r in caplog.records)

    @patch("transcriber._retry.time.sleep")
    def test_http_status_logged_in_warning(
        self, mock_sleep: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """HTTP status code is included in the per-attempt warning log."""
        resp = MagicMock()
        resp.status_code = 503
        resp.headers = {}
        exc = requests.exceptions.HTTPError(response=resp)

        fn = MagicMock(side_effect=[exc, "ok"])
        with caplog.at_level("WARNING", logger="transcriber._retry"):
            retry_with_backoff(
                fn,
                max_retries=2,
                exceptions=(requests.exceptions.RequestException,),
                operation_name="svc",
            )

        assert any("HTTP 503" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Integration-style tests with mock HTTP
# ---------------------------------------------------------------------------


class TestRetryIntegration:
    """Integration tests verifying retry in AzureTranscriptionBackend."""

    @patch("transcriber._retry.time.sleep")
    @patch("requests.post")
    def test_transcription_retries_on_503(
        self, mock_post: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """503 errors are retried; success on 3rd attempt."""
        from transcriber._transcription import AzureTranscriptionBackend

        # First two calls → 503, third → success
        resp_fail = MagicMock()
        resp_fail.status_code = 503
        resp_fail.headers = {}
        resp_fail.text = "Service Unavailable"
        resp_fail.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp_fail)

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.raise_for_status.return_value = None
        resp_ok.json.return_value = {"text": "Hello world"}

        mock_post.side_effect = [resp_fail, resp_fail, resp_ok]

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            f.write(b"fake audio")
            f.flush()
            backend = AzureTranscriptionBackend("key", "https://example.com", max_retries=3)
            result = backend.transcribe(f.name)

        assert result == "Hello world"
        assert mock_post.call_count == 3

    @patch("transcriber._retry.time.sleep")
    @patch("requests.post")
    def test_transcription_no_retry_on_401(
        self, mock_post: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """401 errors are not retried (permanent)."""
        from transcriber._transcription import AzureTranscriptionBackend

        resp = MagicMock()
        resp.status_code = 401
        resp.headers = {}
        resp.text = "Unauthorized"
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)

        mock_post.return_value = resp

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as f:
            f.write(b"fake audio")
            f.flush()
            backend = AzureTranscriptionBackend("key", "https://example.com", max_retries=3)
            with pytest.raises(TranscriptionError) as exc_info:
                backend.transcribe(f.name)

        assert exc_info.value.status_code == 401
        assert mock_post.call_count == 1  # No retry

    @patch("transcriber._retry.time.sleep")
    @patch("requests.post")
    def test_llm_retries_on_429(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        """429 errors are retried with Retry-After support."""
        from transcriber._llm import AzureLLMBackend

        resp_fail = MagicMock()
        resp_fail.status_code = 429
        resp_fail.headers = {"Retry-After": "5"}
        resp_fail.text = "Too Many Requests"
        resp_fail.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp_fail)

        resp_ok = MagicMock()
        resp_ok.status_code = 200
        resp_ok.raise_for_status.return_value = None
        resp_ok.json.return_value = {"choices": [{"message": {"content": "corrected text"}}]}

        mock_post.side_effect = [resp_fail, resp_ok]

        backend = AzureLLMBackend("key", "https://example.com")
        result = backend.complete("test prompt", max_retries=3)

        assert result == "corrected text"
        assert mock_post.call_count == 2

    @patch("transcriber._retry.time.sleep")
    @patch("requests.post")
    def test_llm_no_retry_on_400(self, mock_post: MagicMock, mock_sleep: MagicMock) -> None:
        """400 errors are not retried."""
        from transcriber._llm import AzureLLMBackend

        resp = MagicMock()
        resp.status_code = 400
        resp.headers = {}
        resp.text = "Bad Request"
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(response=resp)

        mock_post.return_value = resp

        backend = AzureLLMBackend("key", "https://example.com")
        with pytest.raises(LLMError) as exc_info:
            backend.complete("test prompt", max_retries=3)

        assert exc_info.value.status_code == 400
        assert mock_post.call_count == 1


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
