"""
Tests for core transcription functionality using real Azure API calls.
"""

import re


class TestTranscription:
    """Tests for Azure transcription API integration."""

    def test_transcribe_short_audio(self, short_audio_file, azure_transcription_backend):
        """Basic transcription returns text with timestamps."""
        result = azure_transcription_backend.transcribe(short_audio_file)

        assert result, "Transcription returned empty result"
        assert len(result) > 0

        # Should contain timestamp formatting [X.XXs - Y.YYs]
        assert re.search(r"\[\d+\.\d+s - \d+\.\d+s\]", result), (
            "Transcription should contain timestamp formatting"
        )

    def test_transcribe_returns_diarized_output(
        self, short_audio_file, azure_transcription_backend
    ):
        """Output contains speaker labels from diarization."""
        result = azure_transcription_backend.transcribe(short_audio_file)

        # Should contain speaker labels (e.g., "A:", "B:", etc.)
        assert re.search(r"\] [A-Z]:", result), "Transcription should contain speaker labels"

    def test_transcribe_multi_speaker_audio(self, multi_speaker_audio, azure_transcription_backend):
        """Multiple speakers are identified in multi-speaker audio."""
        result = azure_transcription_backend.transcribe(multi_speaker_audio)

        # Find all unique speaker labels (A, B, C, etc.)
        speakers = set(re.findall(r"\] ([A-Z]):", result))

        assert len(speakers) >= 2, f"Expected at least 2 speakers, found: {speakers}"

    def test_transcribe_with_time_offset(self, short_audio_file, azure_transcription_backend):
        """Time offset is correctly applied to timestamps."""
        offset = 900  # 15 minutes in seconds

        result = azure_transcription_backend.transcribe(short_audio_file, time_offset=offset)

        match = re.search(r"\[(\d+\.\d+)s -", result)
        assert match, "Could not find timestamp in result"

        first_timestamp = float(match.group(1))

        assert first_timestamp >= offset, (
            f"First timestamp {first_timestamp} should be >= offset {offset}"
        )

    def test_transcribe_returns_readable_text(self, short_audio_file, azure_transcription_backend):
        """Transcription contains actual readable words."""
        result = azure_transcription_backend.transcribe(short_audio_file)

        text_only = re.sub(r"\[\d+\.\d+s - \d+\.\d+s\] Speaker \d+: ", "", result)

        words = text_only.split()
        assert len(words) >= 3, f"Expected transcription to contain words, got: {text_only}"
