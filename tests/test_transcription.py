"""
Tests for core transcription functionality using real Azure API calls.
"""

import re

from transcriber import transcribe_audio


class TestTranscription:
    """Tests for Azure transcription API integration."""

    def test_transcribe_short_audio(self, short_audio_file, azure_transcribe_config):
        """Basic transcription returns text with timestamps."""
        result = transcribe_audio(
            short_audio_file,
            azure_transcribe_config["transcribe_key"],
            azure_transcribe_config["transcribe_url"],
        )

        # Result should be non-empty
        assert result, "Transcription returned empty result"
        assert len(result) > 0

        # Should contain timestamp formatting [X.XXs - Y.YYs]
        assert re.search(r"\[\d+\.\d+s - \d+\.\d+s\]", result), (
            "Transcription should contain timestamp formatting"
        )

    def test_transcribe_returns_diarized_output(self, short_audio_file, azure_transcribe_config):
        """Output contains speaker labels from diarization."""
        result = transcribe_audio(
            short_audio_file,
            azure_transcribe_config["transcribe_key"],
            azure_transcribe_config["transcribe_url"],
        )

        # Should contain speaker labels (e.g., "A:", "B:", etc.)
        assert re.search(r"\] [A-Z]:", result), "Transcription should contain speaker labels"

    def test_transcribe_multi_speaker_audio(self, multi_speaker_audio, azure_transcribe_config):
        """Multiple speakers are identified in multi-speaker audio."""
        result = transcribe_audio(
            multi_speaker_audio,
            azure_transcribe_config["transcribe_key"],
            azure_transcribe_config["transcribe_url"],
        )

        # Find all unique speaker labels (A, B, C, etc.)
        speakers = set(re.findall(r"\] ([A-Z]):", result))

        # Should have at least 2 distinct speakers
        assert len(speakers) >= 2, f"Expected at least 2 speakers, found: {speakers}"

    def test_transcribe_with_time_offset(self, short_audio_file, azure_transcribe_config):
        """Time offset is correctly applied to timestamps."""
        offset = 900  # 15 minutes in seconds

        result = transcribe_audio(
            short_audio_file,
            azure_transcribe_config["transcribe_key"],
            azure_transcribe_config["transcribe_url"],
            time_offset=offset,
        )

        # Extract first timestamp
        match = re.search(r"\[(\d+\.\d+)s -", result)
        assert match, "Could not find timestamp in result"

        first_timestamp = float(match.group(1))

        # First timestamp should be >= offset (since audio starts at 0)
        assert first_timestamp >= offset, (
            f"First timestamp {first_timestamp} should be >= offset {offset}"
        )

    def test_transcribe_returns_readable_text(self, short_audio_file, azure_transcribe_config):
        """Transcription contains actual readable words."""
        result = transcribe_audio(
            short_audio_file,
            azure_transcribe_config["transcribe_key"],
            azure_transcribe_config["transcribe_url"],
        )

        # Remove timestamps and speaker labels to get just the text
        text_only = re.sub(r"\[\d+\.\d+s - \d+\.\d+s\] Speaker \d+: ", "", result)

        # Should have some actual content (words)
        words = text_only.split()
        assert len(words) >= 3, f"Expected transcription to contain words, got: {text_only}"
