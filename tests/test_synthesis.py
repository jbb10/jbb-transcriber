"""
Tests for transcript synthesis functionality.
"""

from transcriber import load_synthesis_prompt, synthesise_transcript, transcribe_audio


class TestSynthesisPrompt:
    """Tests for synthesis prompt loading."""

    def test_load_synthesis_prompt(self):
        """Synthesis prompt can be loaded from package data."""
        prompt = load_synthesis_prompt()

        assert prompt, "Prompt should not be empty"
        assert "{{transcript}}" in prompt, "Prompt should contain {{transcript}} placeholder"
        assert "business writer" in prompt.lower(), "Prompt should mention business writer role"

    def test_synthesis_prompt_has_required_sections(self):
        """Synthesis prompt includes expected structure guidance."""
        prompt = load_synthesis_prompt()

        # Check for key structural elements from the prompt
        assert "Executive summary" in prompt, "Prompt should mention executive summary"
        assert "Decisions" in prompt, "Prompt should mention decisions"
        assert "Actions" in prompt, "Prompt should mention actions"


class TestSynthesisGeneration:
    """Tests for synthesis transcript generation."""

    def test_synthesise_transcript_basic(self, azure_text_config):
        """synthesise_transcript generates a synthesis document."""
        test_transcript = (
            """[0.00s - 10.00s] A: Welcome everyone. """
            """Today we need to discuss the Q1 roadmap.
[10.00s - 20.00s] B: I think we should prioritise the API migration first.
[20.00s - 30.00s] A: Agreed. Let's set a deadline for end of March.
[30.00s - 40.00s] B: I'll take ownership of coordinating with the backend team.
[40.00s - 50.00s] A: Perfect. Any risks we should note?
[50.00s - 60.00s] B: The main risk is dependency on the legacy system being deprecated."""
        )

        synthesis = synthesise_transcript(test_transcript, azure_text_config)

        assert synthesis, "Synthesis should not be empty"
        assert len(synthesis) > 100, "Synthesis should have substantial content"

    def test_synthesise_transcript_contains_key_info(self, azure_text_config):
        """Synthesis captures key information from transcript."""
        test_transcript = """[0.00s - 15.00s] A: We've decided to use Python for the new service.
[15.00s - 30.00s] B: Action item: Sarah will set up the repository by Friday."""

        synthesis = synthesise_transcript(test_transcript, azure_text_config)

        # The synthesis should capture the decision and action item
        # (exact wording depends on LLM output, so we check for presence of concepts)
        synthesis_lower = synthesis.lower()
        assert "python" in synthesis_lower or "decision" in synthesis_lower, (
            "Synthesis should reference the Python decision"
        )

    def test_synthesise_full_transcription(self, short_audio_file, azure_text_config):
        """Synthesis works with a real transcription."""
        # First, get a transcription
        transcription = transcribe_audio(
            short_audio_file,
            azure_text_config["transcribe_key"],
            azure_text_config["transcribe_url"],
        )

        # Generate synthesis
        synthesis = synthesise_transcript(transcription, azure_text_config)

        assert synthesis, "Synthesis should not be empty"
        assert len(synthesis) > 50, "Synthesis should have substantial content"
