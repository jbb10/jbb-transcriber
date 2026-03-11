"""
Tests for transcript synthesis functionality.
"""

from transcriber._pipeline import synthesise_transcript
from transcriber._prompts import build_synthesis_prompt


class TestSynthesisPrompt:
    """Tests for synthesis prompt loading."""

    def test_load_synthesis_prompt(self):
        """Synthesis prompt can be loaded and contains the placeholder."""
        prompt = build_synthesis_prompt("dummy")

        assert prompt, "Prompt should not be empty"
        assert "dummy" in prompt, "Prompt should contain the transcript text"

    def test_synthesis_prompt_has_required_sections(self):
        """Synthesis prompt template includes expected structure guidance."""
        # Build prompt with dummy text; the template content around it
        # should include the key structural elements.
        prompt = build_synthesis_prompt("")

        assert "Executive summary" in prompt, "Prompt should mention executive summary"
        assert "Decisions" in prompt, "Prompt should mention decisions"
        assert "Actions" in prompt, "Prompt should mention actions"


class TestSynthesisGeneration:
    """Tests for synthesis transcript generation."""

    async def test_synthesise_transcript_basic(self, azure_llm_backend):
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

        synthesis = await synthesise_transcript(test_transcript, llm_backend=azure_llm_backend)

        assert synthesis, "Synthesis should not be empty"
        assert len(synthesis) > 100, "Synthesis should have substantial content"

    async def test_synthesise_transcript_contains_key_info(self, azure_llm_backend):
        """Synthesis captures key information from transcript."""
        test_transcript = """[0.00s - 15.00s] A: We've decided to use Python for the new service.
[15.00s - 30.00s] B: Action item: Sarah will set up the repository by Friday."""

        synthesis = await synthesise_transcript(test_transcript, llm_backend=azure_llm_backend)

        synthesis_lower = synthesis.lower()
        assert "python" in synthesis_lower or "decision" in synthesis_lower, (
            "Synthesis should reference the Python decision"
        )

    async def test_synthesise_full_transcription(
        self, short_audio_file, azure_transcription_backend, azure_llm_backend
    ):
        """Synthesis works with a real transcription."""
        transcription = await azure_transcription_backend.transcribe(short_audio_file)

        synthesis = await synthesise_transcript(transcription, llm_backend=azure_llm_backend)

        assert synthesis, "Synthesis should not be empty"
        assert len(synthesis) > 50, "Synthesis should have substantial content"
