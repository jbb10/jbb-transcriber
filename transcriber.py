#!/usr/bin/env python3
"""
Audio Transcription CLI Tool
Transcribes audio files using Azure OpenAI
"""

from __future__ import annotations

__version__ = "0.1.0"

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from types import ModuleType

import av
import requests

# =============================================================================
# Custom argparse types
# =============================================================================


def positive_int(value: str) -> int:
    """Argparse type for positive integers (>= 1)."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: '{value}'") from None
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return ivalue


# =============================================================================
# Validated Configuration
# =============================================================================


@dataclass
class ValidatedConfig:
    """Validated configuration for transcription.

    All fields are validated before the dataclass is created.
    New parameters should be added here and validated in validate_config().
    """

    audio_file: Path
    output_file: Path
    glossary: Path | None
    glossary_text: str | None
    synthesise: bool
    parallel_workers: int
    # Environment variables
    transcribe_api_key: str
    transcribe_url: str
    text_api_key: str | None
    text_url: str | None
    # Local mode configuration
    local_mode: bool
    whisper_model: str

    def as_api_config(self) -> dict[str, str]:
        """Return API credentials as a dict for backward compatibility."""
        config: dict[str, str] = {
            "transcribe_key": self.transcribe_api_key,
            "transcribe_url": self.transcribe_url,
        }
        if self.text_api_key:
            config["text_key"] = self.text_api_key
        if self.text_url:
            config["text_url"] = self.text_url
        return config


def _probe_audio_stream(file_path: Path) -> tuple[bool, str | None]:
    """Check if a file has an audio stream using PyAV.

    Returns:
        Tuple of (has_audio_stream, error_message).
        If has_audio_stream is True, error_message is None.
    """
    try:
        with av.open(str(file_path)) as container:
            for stream in container.streams:
                if stream.type == "audio":
                    return True, None
            return False, "No audio stream found in file"
    except av.error.FileNotFoundError:  # type: ignore[attr-defined]
        return False, f"File not found: {file_path}"
    except av.error.InvalidDataError:  # type: ignore[attr-defined]
        return False, f"Invalid or corrupted media file: {file_path}"
    except Exception as e:
        return False, f"Could not read media file: {e}"


def validate_config(args: argparse.Namespace) -> ValidatedConfig:
    """Validate all CLI arguments and environment variables.

    This function collects ALL validation errors before exiting,
    so users can fix everything at once.

    Args:
        args: Parsed argparse namespace

    Returns:
        ValidatedConfig with all validated values

    Raises:
        SystemExit: If any validation fails
    """
    errors: list[str] = []

    # Derive output path if not provided
    if args.output_file is None:
        output_file = Path(args.audio_file).with_suffix(".txt")
    else:
        output_file = Path(args.output_file)

    audio_file = Path(args.audio_file)

    # --- Validate parallel_workers ---
    if args.parallel_workers > 100:
        errors.append(f"--parallel-workers cannot exceed 100, got {args.parallel_workers}")

    # --- Validate audio file exists ---
    if not audio_file.exists():
        errors.append(f"Audio file not found: {audio_file}")
    elif not audio_file.is_file():
        errors.append(f"Audio path is not a file: {audio_file}")
    else:
        # --- Validate audio file has audio stream ---
        has_audio, audio_error = _probe_audio_stream(audio_file)
        if not has_audio:
            errors.append(audio_error or f"No audio stream in: {audio_file}")

    # --- Validate output directory exists and is writable ---
    output_dir = output_file.parent
    if output_dir and str(output_dir) != ".":
        if not output_dir.exists():
            errors.append(f"Output directory does not exist: {output_dir}")
        elif not os.access(output_dir, os.W_OK):
            errors.append(f"Output directory is not writable: {output_dir}")

    # --- Validate glossary file if provided ---
    glossary_path: Path | None = None
    glossary_text: str | None = None
    if args.glossary:
        glossary_path = Path(args.glossary)
        if not glossary_path.exists():
            errors.append(f"Glossary file not found: {glossary_path}")
        elif not glossary_path.is_file():
            errors.append(f"Glossary path is not a file: {glossary_path}")

    # --- Validate environment variables ---
    transcribe_api_key = os.getenv("AZURE_TRANSCRIBE_API_KEY")
    transcribe_url = os.getenv("AZURE_TRANSCRIBE_URL")
    text_api_key: str | None = None
    text_url: str | None = None

    # Azure transcription credentials only needed when NOT using local mode
    if not args.local:
        if not transcribe_api_key:
            errors.append(
                "AZURE_TRANSCRIBE_API_KEY environment variable is not set. "
                'Add to ~/.zshrc: export AZURE_TRANSCRIBE_API_KEY="your-api-key"'
            )
        if not transcribe_url:
            errors.append(
                "AZURE_TRANSCRIBE_URL environment variable is not set. "
                'Add to ~/.zshrc: export AZURE_TRANSCRIBE_URL="your-endpoint-url"'
            )
    else:
        # In local mode, default to empty strings if not set
        if not transcribe_api_key:
            transcribe_api_key = ""
        if not transcribe_url:
            transcribe_url = ""

    # Text API required if glossary or synthesise is used
    require_text_api = bool(args.glossary) or args.synthesise

    if require_text_api:
        text_api_key = os.getenv("AZURE_TEXT_API_KEY")
        text_url = os.getenv("AZURE_TEXT_URL")

        feature = "--glossary" if args.glossary else "--synthesise"

        if not text_api_key:
            errors.append(
                f"AZURE_TEXT_API_KEY environment variable is not set (required for {feature}). "
                'Add to ~/.zshrc: export AZURE_TEXT_API_KEY="your-api-key"'
            )

        if not text_url:
            errors.append(
                f"AZURE_TEXT_URL environment variable is not set (required for {feature}). "
                'Add to ~/.zshrc: export AZURE_TEXT_URL="your-endpoint-url"'
            )

    # --- Validate Whisper availability in local mode ---
    if args.local:
        try:
            import whisper  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            errors.append(
                "openai-whisper is not installed (required for --local mode). "
                "Reinstall with: uv tool install transcriber"
            )

    # --- Exit if any errors ---
    if errors:
        for err in errors:
            log(f"Error: {err}")
        sys.exit(1)

    # --- Load glossary text (after validation passed) ---
    if glossary_path:
        try:
            glossary_text = glossary_path.read_text(encoding="utf-8")
            log(f"Loaded glossary from: {glossary_path}")
        except OSError as e:
            log(f"Error: Could not read glossary file: {e}")
            sys.exit(1)

    # Log audio file info
    _log_audio_file_info(audio_file)

    return ValidatedConfig(
        audio_file=audio_file,
        output_file=output_file,
        glossary=glossary_path,
        glossary_text=glossary_text,
        synthesise=args.synthesise,
        parallel_workers=args.parallel_workers,
        transcribe_api_key=transcribe_api_key,  # type: ignore[arg-type]
        transcribe_url=transcribe_url,  # type: ignore[arg-type]
        text_api_key=text_api_key,
        text_url=text_url,
        local_mode=args.local,
        whisper_model=args.model,
    )


def _log_audio_file_info(file_path: Path) -> None:
    """Log information about the audio file format."""
    # Formats supported by the API
    api_supported = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
    # Additional audio formats we can convert
    audio_convertible = {".aac", ".ogg", ".flac", ".wma", ".opus", ".aiff", ".aif"}
    # Video formats we can extract audio from
    video_convertible = {".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".3gp", ".mts", ".m2ts"}

    file_ext = file_path.suffix.lower()

    if file_ext in api_supported:
        log(f"File format {file_ext} is directly supported by the API")
    elif file_ext in audio_convertible:
        log(f"Audio file {file_ext} will be converted to a supported format")
    elif file_ext in video_convertible or file_ext == ".mp4":
        log(f"Video file {file_ext} detected - audio will be extracted")
    else:
        log(f"File extension '{file_ext}' is not recognized.")
        log("Attempting to convert anyway...")
        log(f"API supports: {', '.join(sorted(api_supported))}")


def log(message: str) -> None:
    """Print timestamped log message to stderr."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr)


def import_whisper() -> ModuleType:
    """Lazily import Whisper, with helpful error if not installed."""
    try:
        import whisper  # type: ignore[import-not-found]

        return whisper
    except ImportError:
        log("Error: openai-whisper is not installed.")
        log("Reinstall with: uv tool install transcriber")
        log("  or: pip install openai-whisper")
        sys.exit(1)


def format_whisper_output(segments: list[dict[str, object]], time_offset: int = 0) -> str:
    """Format Whisper transcription segments into timestamped text.

    Args:
        segments: List of Whisper segment dicts with 'start', 'end', 'text' keys
        time_offset: Offset in seconds to add to all timestamps (for chunked files)

    Returns:
        Formatted transcription with timestamps
    """
    output_lines: list[str] = []
    for segment in segments:
        start_val = segment.get("start", 0)
        end_val = segment.get("end", 0)
        start = float(start_val) if start_val is not None else 0.0  # type: ignore[arg-type]
        end = float(end_val) if end_val is not None else 0.0  # type: ignore[arg-type]
        start += time_offset
        end += time_offset
        text = str(segment.get("text", "")).strip()
        if text:
            output_lines.append(f"[{start:.2f}s - {end:.2f}s] {text}")
    return "\n".join(output_lines)


def transcribe_audio_local(
    audio_file_path: str, config: ValidatedConfig, time_offset: int = 0
) -> str:
    """Transcribe audio using local Whisper model.

    Args:
        audio_file_path: Path to the audio file
        config: Validated configuration with whisper_model
        time_offset: Offset in seconds to add to all timestamps (for chunked files)

    Returns:
        Formatted transcription text with timestamps
    """
    whisper = import_whisper()

    log(f"Loading Whisper model '{config.whisper_model}'...")

    # Detect GPU availability (torch is a dependency of openai-whisper)
    try:
        import torch  # type: ignore[import-not-found]

        device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[union-attr,unknown-member-type]
    except ImportError:
        device = "cpu"
    log(f"Using device: {device}")

    model = whisper.load_model(config.whisper_model, device=device)
    log("Model loaded successfully")

    log("Transcribing with local Whisper model...")
    log("Language: auto-detect")

    result = model.transcribe(audio_file_path)

    # Log detected language
    if "language" in result:
        log(f"Detected language: {result['language']}")

    segments = result.get("segments", [])
    if not segments:
        # Fallback to plain text if no segments
        return result.get("text", "")

    return format_whisper_output(segments, time_offset)


def get_config(require_text_api: bool = False) -> dict[str, str]:
    """Get configuration from environment variables.

    Args:
        require_text_api: If True, also require text LLM API credentials for glossary correction

    Returns:
        dict with 'transcribe_key', 'transcribe_url', and optionally 'text_key', 'text_url'
    """
    config: dict[str, str] = {}

    api_key = os.getenv("AZURE_TRANSCRIBE_API_KEY")
    api_url = os.getenv("AZURE_TRANSCRIBE_URL")

    if not api_key:
        log("Error: AZURE_TRANSCRIBE_API_KEY environment variable is not set.")
        log("Please add it to your ~/.zshrc file:")
        log('  export AZURE_TRANSCRIBE_API_KEY="your-api-key"')
        sys.exit(1)

    if not api_url:
        log("Error: AZURE_TRANSCRIBE_URL environment variable is not set.")
        log("Please add it to your ~/.zshrc file:")
        log('  export AZURE_TRANSCRIBE_URL="your-endpoint-url"')
        sys.exit(1)

    config["transcribe_key"] = api_key
    config["transcribe_url"] = api_url

    if require_text_api:
        text_key = os.getenv("AZURE_TEXT_API_KEY")
        text_url = os.getenv("AZURE_TEXT_URL")

        if not text_key:
            log("Error: AZURE_TEXT_API_KEY environment variable is not set.")
            log("This is required when using --glossary for transcript correction.")
            log("Please add it to your ~/.zshrc file:")
            log('  export AZURE_TEXT_API_KEY="your-api-key"')
            sys.exit(1)

        if not text_url:
            log("Error: AZURE_TEXT_URL environment variable is not set.")
            log("This is required when using --glossary for transcript correction.")
            log("Please add it to your ~/.zshrc file:")
            log('  export AZURE_TEXT_URL="your-endpoint-url"')
            sys.exit(1)

        config["text_key"] = text_key
        config["text_url"] = text_url

    return config


def validate_audio_file(file_path: str) -> None:
    """Validate that the audio file exists."""
    if not os.path.exists(file_path):
        log(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Formats supported by the API
    api_supported = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
    # Additional audio formats we can convert
    audio_convertible = {".aac", ".ogg", ".flac", ".wma", ".opus", ".aiff", ".aif"}
    # Video formats we can extract audio from
    video_convertible = {".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".3gp", ".mts", ".m2ts"}

    file_ext = Path(file_path).suffix.lower()

    if file_ext in api_supported:
        log(f"File format {file_ext} is directly supported by the API")
    elif file_ext in audio_convertible:
        log(f"Audio file {file_ext} will be converted to a supported format")
    elif file_ext in video_convertible or file_ext == ".mp4":
        log(f"Video file {file_ext} detected - audio will be extracted")
    else:
        log(f"File extension '{file_ext}' is not recognized.")
        log("Attempting to convert anyway...")
        log(f"API supports: {', '.join(sorted(api_supported))}")


def get_audio_duration(file_path: str) -> float | None:
    """Get duration of audio file in seconds using PyAV."""
    try:
        with av.open(file_path) as container:  # type: ignore[arg-type]
            # duration is in time_base units (microseconds for most containers)
            if container.duration is not None:  # type: ignore[union-attr]
                return container.duration / 1_000_000  # type: ignore[union-attr]
            # Fallback: try to get duration from the first audio stream
            for stream in container.streams.audio:  # type: ignore[union-attr]
                if stream.duration is not None:
                    return float(stream.duration * stream.time_base)  # type: ignore[arg-type]
            return None
    except Exception:
        return None


def convert_to_supported_format(file_path: str) -> str:
    """Convert audio/video file to M4A format if needed using PyAV."""
    file_ext = Path(file_path).suffix.lower()
    # API-supported formats that don't need conversion (excluding mp4 which we always convert)
    api_supported_no_conversion = {".mp3", ".mpeg", ".mpga", ".m4a", ".wav"}

    if file_ext in api_supported_no_conversion:
        return file_path

    # Always convert MP4 and WebM to audio-only to reduce file size
    if file_ext in {".mp4", ".webm"}:
        log(f"{file_ext.upper()} file detected - extracting audio track to reduce file size...")
    else:
        log(f"Converting {file_ext} to M4A format...")

    temp_file = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        input_container = av.open(file_path)  # type: ignore[arg-type]
        output_container = av.open(temp_path, mode="w")

        # Find first audio stream
        input_stream = None
        for stream in input_container.streams:
            if stream.type == "audio":
                input_stream = stream
                break

        if input_stream is None:
            log("Error: No audio stream found in file.")
            input_container.close()
            output_container.close()
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            sys.exit(1)

        # Create output stream with AAC codec at 16kHz sample rate
        output_stream = output_container.add_stream("aac", rate=16000)  # type: ignore[union-attr]
        output_stream.bit_rate = 128000  # 128kbps

        # Create resampler to convert to mono 16kHz (optimal for speech recognition)
        resampler = av.AudioResampler(
            format="fltp",  # AAC encoder expects floating point planar
            layout="mono",
            rate=16000,
        )

        # Transcode audio
        for frame in input_container.decode(input_stream):  # type: ignore[arg-type]
            # Resample frame to target format
            resampled_frames = resampler.resample(frame)  # type: ignore[arg-type]
            for resampled_frame in resampled_frames:
                for packet in output_stream.encode(resampled_frame):
                    output_container.mux(packet)

        # Flush the resampler
        for resampled_frame in resampler.resample(None):
            for packet in output_stream.encode(resampled_frame):
                output_container.mux(packet)

        # Flush encoder
        for packet in output_stream.encode():
            output_container.mux(packet)

        input_container.close()
        output_container.close()
        log("Conversion complete")
        return temp_path

    except av.error.FFmpegError as e:  # type: ignore[attr-defined]
        log(f"Error: Could not convert file: {e}")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        sys.exit(1)
    except Exception as e:
        log(f"Error: Unexpected error during conversion: {e}")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        sys.exit(1)


def split_audio_file(file_path: str, chunk_duration: int = 900) -> tuple[list[str], str]:
    """Split audio file into chunks of specified duration (default 15 minutes) using PyAV."""
    log(f"Splitting audio into {chunk_duration // 60}-minute chunks...")

    temp_dir = tempfile.mkdtemp()
    chunks: list[str] = []

    try:
        input_container = av.open(file_path)  # type: ignore[arg-type]

        # Find audio stream
        audio_stream = None
        for stream in input_container.streams:
            if stream.type == "audio":
                audio_stream = stream
                break

        if audio_stream is None:
            log("Error: No audio stream found in file.")
            input_container.close()
            sys.exit(1)

        chunk_index = 0
        chunk_start_time = 0
        output_container = None
        output_stream = None
        resampler = None

        for frame in input_container.decode(audio_stream):  # type: ignore[arg-type]
            # Calculate frame time in seconds
            frame_time = (
                float(frame.pts * frame.time_base)  # type: ignore[operator,arg-type,union-attr]
                if frame.pts is not None
                else 0
            )

            # Check if we need to start a new chunk
            if frame_time >= chunk_start_time + chunk_duration:
                # Flush and close current chunk
                if output_container is not None and output_stream is not None:
                    # Flush resampler
                    if resampler is not None:
                        for resampled_frame in resampler.resample(None):
                            for packet in output_stream.encode(resampled_frame):
                                output_container.mux(packet)
                    # Flush encoder
                    for packet in output_stream.encode():
                        output_container.mux(packet)
                    output_container.close()
                    output_container = None
                    output_stream = None

                # Start new chunk
                chunk_index += 1
                chunk_start_time = chunk_index * chunk_duration

            # Create new output container if needed
            if output_container is None:
                chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index:03d}.m4a")
                chunks.append(chunk_path)
                output_container = av.open(chunk_path, mode="w")
                # Create AAC output stream
                output_stream = output_container.add_stream("aac", rate=16000)  # type: ignore[assignment]
                output_stream.bit_rate = 128000
                # Create resampler for consistent output format
                resampler = av.AudioResampler(
                    format="fltp",
                    layout="mono",
                    rate=16000,
                )

            # Encode frame to output
            if resampler is not None and output_stream is not None:
                resampled_frames = resampler.resample(frame)  # type: ignore[arg-type]
                for resampled_frame in resampled_frames:
                    for packet in output_stream.encode(resampled_frame):
                        output_container.mux(packet)  # type: ignore[union-attr]

        # Close final chunk
        if output_container is not None and output_stream is not None:
            # Flush resampler
            if resampler is not None:
                for resampled_frame in resampler.resample(None):
                    for packet in output_stream.encode(resampled_frame):
                        output_container.mux(packet)
            # Flush encoder
            for packet in output_stream.encode():
                output_container.mux(packet)
            output_container.close()

        input_container.close()

        log(f"Split into {len(chunks)} chunks")
        return chunks, temp_dir

    except av.error.FFmpegError as e:  # type: ignore[attr-defined]
        log(f"Error: Could not split file: {e}")
        sys.exit(1)
    except Exception as e:
        log(f"Error: Unexpected error during split: {e}")
        sys.exit(1)


def transcribe_audio(audio_file_path: str, api_key: str, api_url: str, time_offset: int = 0) -> str:
    """Send audio file to Azure OpenAI API for transcription.

    Args:
        audio_file_path: Path to the audio file
        api_key: Azure API key
        api_url: Azure API endpoint URL
        time_offset: Offset in seconds to add to all timestamps (for chunked files)
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            files = {
                "file": (os.path.basename(audio_file_path), audio_file, "application/octet-stream")
            }

            headers = {"api-key": api_key}

            # For gpt-4o-transcribe-diarize:
            # - response_format can be 'text', 'json', or 'diarized_json'
            # - chunking_strategy is required for audio longer than 30 seconds
            # - 'auto' is recommended for automatic chunking
            data = {
                "model": "gpt-4o-transcribe-diarize",
                "response_format": "diarized_json",
                "chunking_strategy": "auto",
            }

            log("Sending to API for transcription...")

            response = requests.post(
                api_url,
                headers=headers,
                files=files,
                data=data,
                timeout=600,  # 10 minute timeout for large files
            )

            response.raise_for_status()

            result = response.json()

            # Format the output based on response format
            # diarized_json returns: { "text": "...", "segments": [...] }
            # Each segment has: speaker, text, start, end
            if "segments" in result:
                # Format with speaker labels and timestamps
                output_lines: list[str] = []
                for segment in result["segments"]:
                    speaker = segment.get("speaker", "Unknown")
                    text = segment.get("text", "")
                    start = segment.get("start", 0) + time_offset
                    end = segment.get("end", 0) + time_offset
                    output_lines.append(f"[{start:.2f}s - {end:.2f}s] {speaker}: {text}")
                return "\n".join(output_lines)
            elif "text" in result:
                # Fallback to plain text if no segments
                return result["text"]
            else:
                log("Error: Unexpected API response format.")
                log(f"Response: {result}")
                sys.exit(1)

    except requests.exceptions.Timeout:
        log("Error: Request timed out. The audio file may be too large.")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        log(f"Error: API request failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            log(f"Response: {e.response.text}")
        sys.exit(1)
    except OSError as e:
        log(f"Error: Could not read audio file: {e}")
        sys.exit(1)


def load_glossary(glossary_path: str) -> str:
    """Load glossary file content as raw text.

    The glossary can be in any text format - it will be passed directly to the LLM.
    """
    try:
        with open(glossary_path, encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        log(f"Error: Could not read glossary file: {e}")
        sys.exit(1)


def load_correction_prompt() -> str:
    """Load the correction prompt template from package data."""
    try:
        return files("transcriber").joinpath("correction_prompt.md").read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError):
        # Fallback for development: try loading from script directory
        script_dir = Path(__file__).parent
        prompt_path = script_dir / "correction_prompt.md"
        try:
            with open(prompt_path, encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            log(f"Error: Could not read correction prompt file: {e}")
            log(f"Expected at: {prompt_path}")
            sys.exit(1)


def load_synthesis_prompt() -> str:
    """Load the synthesis prompt template from package data."""
    try:
        return files("transcriber").joinpath("synthesis_prompt.md").read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError):
        # Fallback for development: try loading from script directory
        script_dir = Path(__file__).parent
        prompt_path = script_dir / "synthesis_prompt.md"
        try:
            with open(prompt_path, encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            log(f"Error: Could not read synthesis prompt file: {e}")
            log(f"Expected at: {prompt_path}")
            sys.exit(1)


def correct_with_glossary(
    transcript: str, glossary_text: str, config: dict[str, str], max_retries: int = 3
) -> str:
    """Correct transcript using LLM with glossary reference.

    Args:
        transcript: The transcribed text to correct
        glossary_text: The glossary content
        config: Config dict with 'text_key' and 'text_url'
        max_retries: Number of retry attempts with exponential backoff

    Returns:
        Corrected transcript, or original transcript if correction fails
    """
    prompt_template = load_correction_prompt()
    prompt = prompt_template.replace("{{glossary}}", glossary_text).replace(
        "{{transcript}}", transcript
    )

    headers = {"api-key": config["text_key"], "Content-Type": "application/json"}

    data = {
        "model": "gpt-5.1",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Low temperature for consistent corrections
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                backoff = 2**attempt  # 2s, 4s, 8s
                log(f"Retry attempt {attempt + 1}/{max_retries} after {backoff}s backoff...")
                time.sleep(backoff)

            response = requests.post(
                config["text_url"],
                headers=headers,
                json=data,
                timeout=300,  # 5 minute timeout
            )

            response.raise_for_status()
            result = response.json()

            # Extract the corrected text from the response
            if "choices" in result and len(result["choices"]) > 0:
                corrected = result["choices"][0].get("message", {}).get("content", "")
                if corrected.strip():
                    return corrected.strip()

            log("Warning: Unexpected response format from correction API")
            last_error = "Unexpected response format"

        except requests.exceptions.Timeout:
            last_error = "Request timed out"
            log(f"Warning: Correction request timed out (attempt {attempt + 1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            log(f"Warning: Correction request failed (attempt {attempt + 1}/{max_retries}): {e}")

    # All retries failed - fall back to uncorrected transcript
    log(f"Warning: Glossary correction failed after {max_retries} attempts: {last_error}")
    log("Falling back to uncorrected transcript for this segment")
    return transcript


def synthesise_transcript(transcript: str, config: dict[str, str], max_retries: int = 3) -> str:
    """Generate a synthesis document from transcript using LLM.

    Args:
        transcript: The transcribed (and optionally corrected) text
        config: Config dict with 'text_key' and 'text_url'
        max_retries: Number of retry attempts with exponential backoff

    Returns:
        Synthesised markdown document

    Raises:
        RuntimeError: If synthesis fails after all retries
    """
    prompt_template = load_synthesis_prompt()
    prompt = prompt_template.replace("{{transcript}}", transcript)

    headers = {"api-key": config["text_key"], "Content-Type": "application/json"}

    data = {
        "model": "gpt-5.1",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # Slightly higher for more natural writing
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                backoff = 2**attempt  # 2s, 4s, 8s
                log(f"Retry attempt {attempt + 1}/{max_retries} after {backoff}s backoff...")
                time.sleep(backoff)

            response = requests.post(
                config["text_url"],
                headers=headers,
                json=data,
                timeout=300,  # 5 minute timeout
            )

            response.raise_for_status()
            result = response.json()

            # Extract the synthesis from the response
            if "choices" in result and len(result["choices"]) > 0:
                synthesis = result["choices"][0].get("message", {}).get("content", "")
                if synthesis.strip():
                    return synthesis.strip()

            log("Warning: Unexpected response format from synthesis API")
            last_error = "Unexpected response format"

        except requests.exceptions.Timeout:
            last_error = "Request timed out"
            log(f"Warning: Synthesis request timed out (attempt {attempt + 1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            log(f"Warning: Synthesis request failed (attempt {attempt + 1}/{max_retries}): {e}")

    # All retries failed
    raise RuntimeError(f"Synthesis failed after {max_retries} attempts: {last_error}")


def process_chunk(
    chunk_info: tuple[int, str, int], config: dict[str, str], glossary_text: str | None = None
) -> tuple[int, str]:
    """Process a single audio chunk: transcribe and optionally correct.

    Args:
        chunk_info: Tuple of (index, chunk_path, time_offset)
        config: Config dict with API credentials
        glossary_text: Optional glossary content for correction

    Returns:
        Tuple of (index, transcribed_text)
    """
    index, chunk_path, time_offset = chunk_info

    log(f"Transcribing chunk {index + 1}...")
    transcription = transcribe_audio(
        chunk_path, config["transcribe_key"], config["transcribe_url"], time_offset=time_offset
    )

    if glossary_text:
        log(f"Applying glossary correction to chunk {index + 1}...")
        transcription = correct_with_glossary(transcription, glossary_text, config)

    log(f"Chunk {index + 1} complete")
    return (index, transcription)


def write_output(text: str, output_file_path: str) -> None:
    """Write transcription to output file."""
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(text)
        log(f"Transcription saved to: {output_file_path}")
    except OSError as e:
        log(f"Error: Could not write to output file: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using Azure OpenAI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  AZURE_TRANSCRIBE_API_KEY    Your Azure OpenAI API key for transcription
  AZURE_TRANSCRIBE_URL        Your Azure OpenAI endpoint URL for transcription
  AZURE_TEXT_API_KEY          Your Azure OpenAI API key for text LLM
                              (required with --glossary or --synthesise)
  AZURE_TEXT_URL              Your Azure OpenAI endpoint URL for text LLM
                              (required with --glossary or --synthesise)

Example:
  transcribe audio.mp3 transcript.txt
  transcribe podcast.m4a output.txt
  transcribe video.mp4
  (output defaults to video.txt if not specified)

With glossary correction:
  transcribe audio.mp3 transcript.txt --glossary terms.txt
  transcribe long_recording.m4a output.txt --glossary company_terms.txt --parallel-workers 10

With synthesis:
  transcribe meeting.mp4 --synthesise
  transcribe meeting.mp4 --glossary terms.txt --synthesise
  (creates both meeting.txt and meeting_synthesis.md)

Note: Files longer than 25 minutes will be automatically split into chunks.
        """,
    )

    parser.add_argument("audio_file", help="Path to the audio file to transcribe")

    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help="Path to output file (optional - defaults to input with .txt extension)",
    )

    parser.add_argument(
        "--glossary",
        "-g",
        help="Path to glossary file with terms/names/acronyms for transcript correction",
    )

    parser.add_argument(
        "--synthesise",
        "-s",
        action="store_true",
        help="Generate a synthesis document (markdown) summarising the transcript",
    )

    parser.add_argument(
        "--parallel-workers",
        "-p",
        type=positive_int,
        default=15,
        help="Maximum number of parallel workers for processing chunks (default: 15, max: 100)",
    )

    parser.add_argument(
        "--local",
        "-l",
        action="store_true",
        help="Use local Whisper model instead of Azure OpenAI "
        "(requires --model to specify which model)",
    )

    parser.add_argument(
        "--model",
        default="base",
        choices=[
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v1",
            "large-v2",
            "large-v3",
            "turbo",
        ],
        help="Whisper model to use with local mode (default: base). "
        "Valid models: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, "
        "large, large-v1, large-v2, large-v3, turbo",
    )

    args = parser.parse_args()

    # Validate ALL parameters upfront before any work begins
    validated = validate_config(args)
    api_config = validated.as_api_config()

    # Convert to supported format if needed (needed for both local and Azure)
    converted_file = convert_to_supported_format(str(validated.audio_file))
    temp_converted = converted_file != str(validated.audio_file)

    try:
        if validated.local_mode:
            # Local Whisper transcription
            log("Using local Whisper model for transcription")

            try:
                final_transcript = transcribe_audio_local(converted_file, validated)

                # Apply glossary correction if provided (uses Azure LLM)
                if validated.glossary_text:
                    log("Applying glossary correction...")
                    final_transcript = correct_with_glossary(
                        final_transcript, validated.glossary_text, api_config
                    )

                write_output(final_transcript, str(validated.output_file))

                # Generate synthesis if requested (uses Azure LLM)
                if validated.synthesise:
                    output_stem = validated.output_file.with_suffix("")
                    synthesis_output = str(output_stem) + "_synthesis.md"
                    log("Generating synthesis document...")
                    try:
                        synthesis = synthesise_transcript(final_transcript, api_config)
                        write_output(synthesis, synthesis_output)
                    except RuntimeError as e:
                        log(f"Warning: {e}")
                        log(
                            "Transcript was saved successfully, "
                            "but synthesis could not be generated."
                        )
            finally:
                # Local mode doesn't create temp chunks like Azure mode
                pass
        else:
            # Azure API transcription (existing behavior)
            # Check duration and split if necessary
            duration = get_audio_duration(converted_file)
            max_duration = 1400  # 23 minutes 20 seconds (safe margin under 25 min limit)

            if duration and duration > max_duration:
                log(f"Audio duration: {duration / 60:.1f} min (exceeds limit)")
                chunks, temp_dir = split_audio_file(
                    converted_file, chunk_duration=900
                )  # 15-minute chunks

                try:
                    chunk_duration = 900  # 15-minute chunks

                    # Prepare chunk info for parallel processing
                    chunk_infos = [(i, chunk, i * chunk_duration) for i, chunk in enumerate(chunks)]

                    # Process chunks in parallel
                    num_workers = min(validated.parallel_workers, len(chunks))
                    log(f"Processing {len(chunks)} chunks with {num_workers} parallel workers...")

                    results: dict[int, str] = {}
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        # Submit all tasks
                        future_to_index = {
                            executor.submit(
                                process_chunk, info, api_config, validated.glossary_text
                            ): info[0]
                            for info in chunk_infos
                        }

                        # Collect results as they complete
                        for future in as_completed(future_to_index):
                            index = future_to_index[future]
                            try:
                                idx, transcription = future.result()
                                results[idx] = transcription
                                log(f"Completed chunk {idx + 1}/{len(chunks)}")
                            except Exception as e:
                                log(f"Error processing chunk {index + 1}: {e}")
                                sys.exit(1)

                    # Combine all transcriptions in correct order
                    all_transcriptions = [results[i] for i in range(len(chunks))]
                    final_transcript = "\n".join(all_transcriptions)
                    write_output(final_transcript, str(validated.output_file))
                finally:
                    # Clean up chunks
                    log("Cleaning up temporary audio files...")
                    for chunk in chunks:
                        if os.path.exists(chunk):
                            os.unlink(chunk)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                    log("Temporary files cleaned up")
            else:
                # Transcribe single file
                final_transcript = transcribe_audio(
                    converted_file, api_config["transcribe_key"], api_config["transcribe_url"]
                )

                # Apply glossary correction if provided
                if validated.glossary_text:
                    log("Applying glossary correction...")
                    final_transcript = correct_with_glossary(
                        final_transcript, validated.glossary_text, api_config
                    )

                write_output(final_transcript, str(validated.output_file))

            # Generate synthesis if requested
            if validated.synthesise:
                output_stem = validated.output_file.with_suffix("")
                synthesis_output = str(output_stem) + "_synthesis.md"
                log("Generating synthesis document...")
                try:
                    synthesis = synthesise_transcript(final_transcript, api_config)
                    write_output(synthesis, synthesis_output)
                except RuntimeError as e:
                    log(f"Warning: {e}")
                    log("Transcript was saved successfully, but synthesis could not be generated.")
    finally:
        # Clean up converted file if we created one
        if temp_converted and os.path.exists(converted_file):
            log("Cleaning up converted audio file...")
            os.unlink(converted_file)

    log("Transcription complete!")


if __name__ == "__main__":
    main()
