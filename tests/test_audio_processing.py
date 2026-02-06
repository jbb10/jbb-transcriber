"""
Tests for audio processing functions (format conversion, duration, splitting).
"""

import os
import tempfile

import pytest

from transcriber import (
    convert_to_supported_format,
    get_audio_duration,
    split_audio_file,
)


class TestAudioDuration:
    """Tests for audio duration extraction."""

    def test_get_audio_duration_returns_positive_value(self, short_audio_file):
        """Duration extraction returns a positive number for valid audio."""
        duration = get_audio_duration(short_audio_file)

        assert duration is not None, "Duration should not be None for valid audio"
        assert duration > 0, f"Duration should be positive, got {duration}"

    def test_get_audio_duration_reasonable_range(self, short_audio_file):
        """Short audio file duration is in expected range (1-60 seconds)."""
        duration = get_audio_duration(short_audio_file)

        assert duration is not None
        # Short speech should be between 1 second and 60 seconds
        assert 1 <= duration <= 60, f"Short audio duration {duration}s outside expected range 1-60s"

    def test_get_audio_duration_returns_none_for_invalid_file(self, fixtures_dir):
        """Duration extraction returns None for invalid/non-audio files."""
        # Create a fake file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"not a real audio file")
            fake_path = f.name

        try:
            duration = get_audio_duration(fake_path)
            # Should return None or raise an exception gracefully
            assert duration is None, "Duration of invalid file should be None"
        finally:
            os.unlink(fake_path)

    def test_get_audio_duration_multi_speaker(self, multi_speaker_audio):
        """Duration extraction works for multi-speaker audio."""
        duration = get_audio_duration(multi_speaker_audio)

        assert duration is not None, "Duration should not be None"
        assert duration > 0, "Duration should be positive"


class TestFormatConversion:
    """Tests for audio format conversion."""

    def test_convert_to_supported_format_mp3_passthrough(self, short_audio_file):
        """MP3 files are passed through without conversion."""
        result = convert_to_supported_format(short_audio_file)

        # MP3 should not be converted, return same path
        assert result == short_audio_file, "MP3 files should pass through without conversion"

    def test_convert_to_supported_format_creates_valid_m4a(self, fixtures_dir):
        """Conversion to M4A produces a valid audio file."""
        # This test requires a non-MP3 audio file
        # Try to find any convertible file in fixtures
        convertible_extensions = [".ogg", ".flac", ".wav", ".webm", ".mp4"]

        source_file = None
        for ext in convertible_extensions:
            for f in fixtures_dir.glob(f"*{ext}"):
                source_file = str(f)
                break
            if source_file:
                break

        if not source_file:
            pytest.skip("No convertible audio file found in fixtures")

        result = convert_to_supported_format(source_file)

        try:
            # Should return a different path (temporary m4a file)
            assert result != source_file, "Converted file should have different path"

            # Should be an m4a file
            assert result.endswith(".m4a"), f"Converted file should be .m4a, got {result}"

            # File should exist and have content
            assert os.path.exists(result), "Converted file should exist"
            assert os.path.getsize(result) > 0, "Converted file should have content"

            # Should be valid audio (duration should be extractable)
            duration = get_audio_duration(result)
            assert duration is not None, "Converted file should have a duration"
            assert duration > 0, "Converted file should be valid audio"
        finally:
            # Clean up if a temp file was created
            if result != source_file and os.path.exists(result):
                os.unlink(result)


class TestAudioSplitting:
    """Tests for audio file splitting."""

    def test_split_long_audio(self, long_audio_file):
        """Long audio files are split into chunks."""
        chunks, temp_dir = split_audio_file(long_audio_file, chunk_duration=900)

        try:
            # Should have multiple chunks
            assert len(chunks) >= 2, f"Expected at least 2 chunks for long audio, got {len(chunks)}"

            # All chunks should exist
            for chunk in chunks:
                assert os.path.exists(chunk), f"Chunk file should exist: {chunk}"
                assert os.path.getsize(chunk) > 0, f"Chunk should have content: {chunk}"

            # Chunks should be valid audio
            for chunk in chunks:
                duration = get_audio_duration(chunk)
                assert duration is not None, f"Chunk should have valid duration: {chunk}"
                # Each chunk should be roughly 15 minutes or less (with some tolerance)
                assert duration <= 1000, f"Chunk duration {duration}s exceeds expected max ~900s"

        finally:
            # Clean up
            for chunk in chunks:
                if os.path.exists(chunk):
                    os.unlink(chunk)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

    def test_split_returns_chunks_in_order(self, long_audio_file):
        """Split chunks are numbered and returned in order."""
        chunks, temp_dir = split_audio_file(long_audio_file, chunk_duration=900)

        try:
            # Check chunk naming pattern
            for i, chunk in enumerate(chunks):
                expected_name = f"chunk_{i:03d}.m4a"
                assert chunk.endswith(expected_name), (
                    f"Chunk {i} should be named {expected_name}, got {os.path.basename(chunk)}"
                )

        finally:
            for chunk in chunks:
                if os.path.exists(chunk):
                    os.unlink(chunk)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
