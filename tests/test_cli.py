"""
CLI integration tests for the transcribe command.
"""

import os
import re
import subprocess
import sys
import tempfile


class TestCLIBasicUsage:
    """Tests for basic CLI functionality."""

    def test_cli_basic_usage(
        self, short_audio_file, temp_output_file, azure_transcribe_config, monkeypatch
    ):
        """transcribe input.mp3 output.txt works correctly."""
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env["AZURE_TRANSCRIBE_API_KEY"] = azure_transcribe_config["transcribe_key"]
        env["AZURE_TRANSCRIBE_URL"] = azure_transcribe_config["transcribe_url"]

        result = subprocess.run(
            [sys.executable, "-m", "transcriber", short_audio_file, temp_output_file],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        # Output file should exist and have content
        assert os.path.exists(temp_output_file), "Output file should be created"
        with open(temp_output_file, encoding="utf-8") as f:
            content = f.read()

        assert len(content) > 0, "Output file should have content"
        # Should contain speaker labels (e.g., "A:", "B:")
        assert re.search(r"\] [A-Z]:", content), "Output should contain speaker labels"

    def test_cli_default_output(self, short_audio_file, azure_transcribe_config):
        """Output defaults to input filename with .txt extension."""
        env = os.environ.copy()
        env["AZURE_TRANSCRIBE_API_KEY"] = azure_transcribe_config["transcribe_key"]
        env["AZURE_TRANSCRIBE_URL"] = azure_transcribe_config["transcribe_url"]

        # Copy audio to temp location to avoid polluting fixtures
        import shutil

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio = os.path.join(temp_dir, "test_audio.mp3")
            shutil.copy(short_audio_file, temp_audio)

            expected_output = os.path.join(temp_dir, "test_audio.txt")

            result = subprocess.run(
                [sys.executable, "-m", "transcriber", temp_audio],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )

            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(expected_output), (
                f"Default output file {expected_output} should be created"
            )


class TestCLIWithGlossary:
    """Tests for CLI with glossary correction."""

    def test_cli_with_glossary(
        self, short_audio_file, temp_output_file, azure_text_config, sample_glossary
    ):
        """--glossary flag works correctly."""
        env = os.environ.copy()
        env["AZURE_TRANSCRIBE_API_KEY"] = azure_text_config["transcribe_key"]
        env["AZURE_TRANSCRIBE_URL"] = azure_text_config["transcribe_url"]
        env["AZURE_TEXT_API_KEY"] = azure_text_config["text_key"]
        env["AZURE_TEXT_URL"] = azure_text_config["text_url"]

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "transcriber",
                short_audio_file,
                temp_output_file,
                "--glossary",
                sample_glossary,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for transcription + correction
        )

        assert result.returncode == 0, f"CLI with glossary failed: {result.stderr}"

        # Output file should exist and have content
        assert os.path.exists(temp_output_file)
        with open(temp_output_file, encoding="utf-8") as f:
            content = f.read()

        assert len(content) > 0

    def test_cli_glossary_short_flag(
        self, short_audio_file, temp_output_file, azure_text_config, sample_glossary
    ):
        """-g short flag for glossary works."""
        env = os.environ.copy()
        env["AZURE_TRANSCRIBE_API_KEY"] = azure_text_config["transcribe_key"]
        env["AZURE_TRANSCRIBE_URL"] = azure_text_config["transcribe_url"]
        env["AZURE_TEXT_API_KEY"] = azure_text_config["text_key"]
        env["AZURE_TEXT_URL"] = azure_text_config["text_url"]

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "transcriber",
                short_audio_file,
                temp_output_file,
                "-g",
                sample_glossary,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )

        assert result.returncode == 0, f"CLI with -g flag failed: {result.stderr}"


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_cli_missing_input_file(self):
        """CLI fails gracefully for missing input file."""
        result = subprocess.run(
            [sys.executable, "-m", "transcriber", "/nonexistent/file.mp3", "output.txt"],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Should fail for missing input file"

    def test_cli_missing_glossary_file(
        self, short_audio_file, temp_output_file, azure_transcribe_config
    ):
        """CLI fails gracefully for missing glossary file."""
        env = os.environ.copy()
        env["AZURE_TRANSCRIBE_API_KEY"] = azure_transcribe_config["transcribe_key"]
        env["AZURE_TRANSCRIBE_URL"] = azure_transcribe_config["transcribe_url"]

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "transcriber",
                short_audio_file,
                temp_output_file,
                "--glossary",
                "/nonexistent/glossary.txt",
            ],
            env=env,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Should fail for missing glossary file"

    def test_cli_help(self):
        """CLI --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "transcriber", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "transcribe" in result.stdout.lower() or "audio" in result.stdout.lower()
        assert "--glossary" in result.stdout
        assert "--synthesise" in result.stdout

    def test_cli_parallel_workers_option(
        self, short_audio_file, temp_output_file, azure_transcribe_config
    ):
        """--parallel-workers option is accepted."""
        env = os.environ.copy()
        env["AZURE_TRANSCRIBE_API_KEY"] = azure_transcribe_config["transcribe_key"]
        env["AZURE_TRANSCRIBE_URL"] = azure_transcribe_config["transcribe_url"]

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "transcriber",
                short_audio_file,
                temp_output_file,
                "--parallel-workers",
                "5",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, f"CLI with --parallel-workers failed: {result.stderr}"


class TestCLIWithSynthesis:
    """Tests for CLI with synthesis generation."""

    def test_cli_with_synthesise(self, short_audio_file, temp_output_file, azure_text_config):
        """--synthesise flag generates synthesis document."""
        import shutil

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio = os.path.join(temp_dir, "test_audio.mp3")
            shutil.copy(short_audio_file, temp_audio)

            transcript_output = os.path.join(temp_dir, "test_audio.txt")
            expected_synthesis = os.path.join(temp_dir, "test_audio_synthesis.md")

            env = os.environ.copy()
            env["AZURE_TRANSCRIBE_API_KEY"] = azure_text_config["transcribe_key"]
            env["AZURE_TRANSCRIBE_URL"] = azure_text_config["transcribe_url"]
            env["AZURE_TEXT_API_KEY"] = azure_text_config["text_key"]
            env["AZURE_TEXT_URL"] = azure_text_config["text_url"]

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "transcriber",
                    temp_audio,
                    "--synthesise",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )

            assert result.returncode == 0, f"CLI with --synthesise failed: {result.stderr}"

            # Both transcript and synthesis should exist
            assert os.path.exists(transcript_output), "Transcript file should be created"
            assert os.path.exists(expected_synthesis), "Synthesis file should be created"

            # Synthesis should have content
            with open(expected_synthesis, encoding="utf-8") as f:
                synthesis_content = f.read()
            assert len(synthesis_content) > 0, "Synthesis file should have content"

    def test_cli_synthesise_short_flag(self, short_audio_file, azure_text_config):
        """-s short flag for synthesise works."""
        import shutil

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio = os.path.join(temp_dir, "test_audio.mp3")
            shutil.copy(short_audio_file, temp_audio)

            expected_synthesis = os.path.join(temp_dir, "test_audio_synthesis.md")

            env = os.environ.copy()
            env["AZURE_TRANSCRIBE_API_KEY"] = azure_text_config["transcribe_key"]
            env["AZURE_TRANSCRIBE_URL"] = azure_text_config["transcribe_url"]
            env["AZURE_TEXT_API_KEY"] = azure_text_config["text_key"]
            env["AZURE_TEXT_URL"] = azure_text_config["text_url"]

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "transcriber",
                    temp_audio,
                    "-s",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )

            assert result.returncode == 0, f"CLI with -s flag failed: {result.stderr}"
            assert os.path.exists(expected_synthesis), (
                "Synthesis file should be created with -s flag"
            )

    def test_cli_glossary_and_synthesise(
        self, short_audio_file, azure_text_config, sample_glossary
    ):
        """--glossary and --synthesise work together."""
        import shutil

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio = os.path.join(temp_dir, "test_audio.mp3")
            shutil.copy(short_audio_file, temp_audio)

            transcript_output = os.path.join(temp_dir, "test_audio.txt")
            expected_synthesis = os.path.join(temp_dir, "test_audio_synthesis.md")

            env = os.environ.copy()
            env["AZURE_TRANSCRIBE_API_KEY"] = azure_text_config["transcribe_key"]
            env["AZURE_TRANSCRIBE_URL"] = azure_text_config["transcribe_url"]
            env["AZURE_TEXT_API_KEY"] = azure_text_config["text_key"]
            env["AZURE_TEXT_URL"] = azure_text_config["text_url"]

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "transcriber",
                    temp_audio,
                    "--glossary",
                    sample_glossary,
                    "--synthesise",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=900,  # Longer timeout for transcription + correction + synthesis
            )

            assert result.returncode == 0, (
                f"CLI with glossary and synthesise failed: {result.stderr}"
            )

            # Both outputs should exist
            assert os.path.exists(transcript_output), "Transcript file should be created"
            assert os.path.exists(expected_synthesis), "Synthesis file should be created"


class TestCLISynthesiseOnly:
    """Tests for CLI --synthesise-only flag."""

    def test_cli_synthesise_only(self, azure_text_config):
        """--synthesise-only reads existing transcript and generates synthesis."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a transcript file
            transcript_path = os.path.join(temp_dir, "meeting.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(
                    "[0.00s - 10.00s] A: We decided to migrate.\n"
                    "[10.00s - 20.00s] B: John will update CI.\n"
                    "[20.00s - 30.00s] A: Any risks?\n"
                    "[30.00s - 40.00s] B: Backwards compatibility.\n"
                )

            expected_synthesis = os.path.join(temp_dir, "meeting_synthesis.md")

            env = os.environ.copy()
            env["AZURE_TEXT_API_KEY"] = azure_text_config["text_key"]
            env["AZURE_TEXT_URL"] = azure_text_config["text_url"]

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "transcriber",
                    transcript_path,
                    "--synthesise-only",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )

            assert result.returncode == 0, f"CLI with --synthesise-only failed: {result.stderr}"
            assert os.path.exists(expected_synthesis), "Synthesis file should be created"

            with open(expected_synthesis, encoding="utf-8") as f:
                synthesis_content = f.read()
            assert len(synthesis_content) > 0, "Synthesis file should have content"

    def test_cli_synthesise_only_short_flag(self, azure_text_config):
        """-S short flag for synthesise-only works."""
        with tempfile.TemporaryDirectory() as temp_dir:
            transcript_path = os.path.join(temp_dir, "notes.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write("[0.00s - 10.00s] A: Let's discuss the roadmap.\n")

            expected_synthesis = os.path.join(temp_dir, "notes_synthesis.md")

            env = os.environ.copy()
            env["AZURE_TEXT_API_KEY"] = azure_text_config["text_key"]
            env["AZURE_TEXT_URL"] = azure_text_config["text_url"]

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "transcriber",
                    transcript_path,
                    "-S",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )

            assert result.returncode == 0, f"CLI with -S flag failed: {result.stderr}"
            assert os.path.exists(expected_synthesis), (
                "Synthesis file should be created with -S flag"
            )

    def test_cli_synthesise_only_empty_file_fails(self, azure_text_config):
        """--synthesise-only fails on empty transcript file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            transcript_path = os.path.join(temp_dir, "empty.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write("")

            env = os.environ.copy()
            env["AZURE_TEXT_API_KEY"] = azure_text_config["text_key"]
            env["AZURE_TEXT_URL"] = azure_text_config["text_url"]

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "transcriber",
                    transcript_path,
                    "--synthesise-only",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

            assert result.returncode != 0, "Should fail on empty transcript file"
            assert "empty" in result.stderr.lower()

    def test_cli_synthesise_only_missing_file_fails(self):
        """--synthesise-only fails for nonexistent transcript file."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "transcriber",
                "/nonexistent/transcript.txt",
                "--synthesise-only",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Should fail for missing transcript file"

    def test_cli_synthesise_and_synthesise_only_conflict(self):
        """--synthesise and --synthesise-only cannot be used together."""
        with tempfile.TemporaryDirectory() as temp_dir:
            transcript_path = os.path.join(temp_dir, "transcript.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write("Some content")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "transcriber",
                    transcript_path,
                    "--synthesise",
                    "--synthesise-only",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode != 0, (
                "Should fail when both --synthesise and --synthesise-only are used"
            )

    def test_cli_help_shows_synthesise_only(self):
        """CLI --help mentions --synthesise-only."""
        result = subprocess.run(
            [sys.executable, "-m", "transcriber", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--synthesise-only" in result.stdout


class TestCLITextFileAutoDetection:
    """Tests for auto-detecting text files and running synthesis."""

    def test_cli_text_file_auto_synthesis(self, azure_text_config):
        """Passing a .txt file without flags auto-enables synthesis-only mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            transcript_path = os.path.join(temp_dir, "meeting.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(
                    "[0.00s - 10.00s] A: We decided to migrate.\n"
                    "[10.00s - 20.00s] B: John will update CI.\n"
                )

            expected_synthesis = os.path.join(temp_dir, "meeting_synthesis.md")

            env = os.environ.copy()
            env["AZURE_TEXT_API_KEY"] = azure_text_config["text_key"]
            env["AZURE_TEXT_URL"] = azure_text_config["text_url"]

            result = subprocess.run(
                [sys.executable, "-m", "transcriber", transcript_path],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )

            assert result.returncode == 0, f"CLI text auto-detect failed: {result.stderr}"
            assert os.path.exists(expected_synthesis), "Synthesis file should be created"
            assert "Text file detected" in result.stderr, "Should notify user about auto-detection"

    def test_cli_text_file_notification_shown(self):
        """Text file auto-detection prints a visible notification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            transcript_path = os.path.join(temp_dir, "notes.md")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write("Some notes")

            env = os.environ.copy()
            env["AZURE_TEXT_API_KEY"] = "test-key"
            env["AZURE_TEXT_URL"] = "https://test.example.com"

            result = subprocess.run(
                [sys.executable, "-m", "transcriber", transcript_path],
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Will fail at synthesis (fake creds), but notification should appear
            assert "Text file detected" in result.stderr
