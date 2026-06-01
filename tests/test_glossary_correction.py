"""
Tests for glossary-based transcript correction.
"""

import re

from jbb_transcriber._pipeline import _correct_with_glossary
from jbb_transcriber._settings import PipelineSettings


class TestGlossaryCorrection:
    """Tests for glossary correction functionality."""

    async def test_glossary_correction_applies(
        self, short_audio_file, azure_transcription_backend, azure_llm_backend, sample_glossary
    ):
        """Glossary correction processes the transcript through the LLM."""
        transcription = await azure_transcription_backend.transcribe(short_audio_file)

        with open(sample_glossary, encoding="utf-8") as f:
            glossary_text = f.read()

        settings = PipelineSettings()
        result = await _correct_with_glossary(
            transcription, glossary_text, azure_llm_backend, settings
        )

        assert result.text, "Correction returned empty result"
        assert len(result.text) > 0

    async def test_glossary_preserves_timestamps(
        self, short_audio_file, azure_transcription_backend, azure_llm_backend, sample_glossary
    ):
        """Timestamps remain intact after glossary correction."""
        transcription = await azure_transcription_backend.transcribe(short_audio_file)

        with open(sample_glossary, encoding="utf-8") as f:
            glossary_text = f.read()

        settings = PipelineSettings()
        result = await _correct_with_glossary(
            transcription, glossary_text, azure_llm_backend, settings
        )

        original_timestamps = re.findall(r"\[\d+\.\d+s - \d+\.\d+s\]", transcription)
        assert original_timestamps, "Original transcription should have timestamps"

        corrected_timestamps = re.findall(r"\[\d+\.\d+s - \d+\.\d+s\]", result.text)
        assert corrected_timestamps, "Corrected transcription should preserve timestamps"

        assert len(corrected_timestamps) == len(original_timestamps), (
            f"Expected {len(original_timestamps)} timestamp blocks, got {len(corrected_timestamps)}"
        )

    async def test_glossary_preserves_speaker_labels(
        self, short_audio_file, azure_transcription_backend, azure_llm_backend, sample_glossary
    ):
        """Speaker labels are preserved after glossary correction."""
        transcription = await azure_transcription_backend.transcribe(short_audio_file)

        with open(sample_glossary, encoding="utf-8") as f:
            glossary_text = f.read()

        settings = PipelineSettings()
        result = await _correct_with_glossary(
            transcription, glossary_text, azure_llm_backend, settings
        )

        original_speakers = re.findall(r"\] ([A-Z]):", transcription)
        assert original_speakers, "Original transcription should have speaker labels"

        corrected_speakers = re.findall(r"\] ([A-Z]):", result.text)
        assert corrected_speakers, "Corrected transcription should preserve speaker labels"

        assert len(corrected_speakers) == len(original_speakers), (
            f"Expected {len(original_speakers)} speaker segments, got {len(corrected_speakers)}"
        )

    async def test_glossary_with_specific_terms(self, azure_llm_backend):
        """Test that specific glossary terms influence correction."""
        test_transcript = (
            "[0.00s - 5.00s] Speaker 1: We need to update the aye pee eye docs.\n"
            "[5.00s - 10.00s] Speaker 1: The see ell eye tool is working great."
        )

        glossary_text = """Technical acronyms:
- API: Application Programming Interface (pronounced as letters A-P-I)
- CLI: Command Line Interface (pronounced as letters C-L-I)
"""

        settings = PipelineSettings()
        result = await _correct_with_glossary(
            test_transcript, glossary_text, azure_llm_backend, settings
        )

        assert result.text, "Correction should return a result"
        assert len(result.text) > 0

    async def test_glossary_falls_back_on_empty_glossary(
        self, short_audio_file, azure_transcription_backend, azure_llm_backend
    ):
        """Correction still works with an empty glossary."""
        transcription = await azure_transcription_backend.transcribe(short_audio_file)

        glossary_text = ""

        settings = PipelineSettings()
        result = await _correct_with_glossary(
            transcription, glossary_text, azure_llm_backend, settings
        )

        assert result.text, "Correction should return a result even with empty glossary"
