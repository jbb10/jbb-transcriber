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
