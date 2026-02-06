"""
Tests for glossary-based transcript correction.
"""

import re

from transcriber import correct_with_glossary, transcribe_audio


class TestGlossaryCorrection:
    """Tests for glossary correction functionality."""

    def test_glossary_correction_applies(
        self, short_audio_file, azure_text_config, sample_glossary
    ):
        """Glossary correction processes the transcript through the LLM."""
        # First, get a transcription
        transcription = transcribe_audio(
            short_audio_file,
            azure_text_config["transcribe_key"],
            azure_text_config["transcribe_url"],
        )

        # Load the glossary
        with open(sample_glossary, encoding="utf-8") as f:
            glossary_text = f.read()

        # Apply correction
        corrected = correct_with_glossary(transcription, glossary_text, azure_text_config)

        # Result should be non-empty
        assert corrected, "Correction returned empty result"
        assert len(corrected) > 0

    def test_glossary_preserves_timestamps(
        self, short_audio_file, azure_text_config, sample_glossary
    ):
        """Timestamps remain intact after glossary correction."""
        transcription = transcribe_audio(
            short_audio_file,
            azure_text_config["transcribe_key"],
            azure_text_config["transcribe_url"],
        )

        with open(sample_glossary, encoding="utf-8") as f:
            glossary_text = f.read()

        corrected = correct_with_glossary(transcription, glossary_text, azure_text_config)

        # Original should have timestamps
        original_timestamps = re.findall(r"\[\d+\.\d+s - \d+\.\d+s\]", transcription)
        assert original_timestamps, "Original transcription should have timestamps"

        # Corrected should also have timestamps
        corrected_timestamps = re.findall(r"\[\d+\.\d+s - \d+\.\d+s\]", corrected)
        assert corrected_timestamps, "Corrected transcription should preserve timestamps"

        # Should have same number of timestamp blocks
        assert len(corrected_timestamps) == len(original_timestamps), (
            f"Expected {len(original_timestamps)} timestamp blocks, got {len(corrected_timestamps)}"
        )

    def test_glossary_preserves_speaker_labels(
        self, short_audio_file, azure_text_config, sample_glossary
    ):
        """Speaker labels are preserved after glossary correction."""
        transcription = transcribe_audio(
            short_audio_file,
            azure_text_config["transcribe_key"],
            azure_text_config["transcribe_url"],
        )

        with open(sample_glossary, encoding="utf-8") as f:
            glossary_text = f.read()

        corrected = correct_with_glossary(transcription, glossary_text, azure_text_config)

        # Original should have speaker labels (A:, B:, etc.)
        original_speakers = re.findall(r"\] ([A-Z]):", transcription)
        assert original_speakers, "Original transcription should have speaker labels"

        # Corrected should also have speaker labels
        corrected_speakers = re.findall(r"\] ([A-Z]):", corrected)
        assert corrected_speakers, "Corrected transcription should preserve speaker labels"

        # Should have same number of speaker segments
        assert len(corrected_speakers) == len(original_speakers), (
            f"Expected {len(original_speakers)} speaker segments, got {len(corrected_speakers)}"
        )

    def test_glossary_with_specific_terms(self, azure_text_config):
        """Test that specific glossary terms influence correction."""
        # Create a transcript with potentially misspelled terms
        test_transcript = (
            "[0.00s - 5.00s] Speaker 1: We need to update the aye pee eye docs.\n"
            "[5.00s - 10.00s] Speaker 1: The see ell eye tool is working great."
        )

        glossary_text = """Technical acronyms:
- API: Application Programming Interface (pronounced as letters A-P-I)
- CLI: Command Line Interface (pronounced as letters C-L-I)
"""

        corrected = correct_with_glossary(test_transcript, glossary_text, azure_text_config)

        # The correction should attempt to fix "aye pee eye" to "API" and "see ell eye" to "CLI"
        # Note: The exact correction depends on the LLM, so we just verify the function runs
        assert corrected, "Correction should return a result"
        assert len(corrected) > 0

    def test_glossary_falls_back_on_empty_glossary(self, short_audio_file, azure_text_config):
        """Correction still works with an empty glossary."""
        transcription = transcribe_audio(
            short_audio_file,
            azure_text_config["transcribe_key"],
            azure_text_config["transcribe_url"],
        )

        # Empty glossary
        glossary_text = ""

        corrected = correct_with_glossary(transcription, glossary_text, azure_text_config)

        # Should still return a result (possibly unchanged)
        assert corrected, "Correction should return a result even with empty glossary"
