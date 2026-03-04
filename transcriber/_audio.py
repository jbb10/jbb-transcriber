"""Audio processing: duration, format conversion, and splitting.

All functions raise typed exceptions instead of calling sys.exit().
Temporary files are managed via context managers for reliable cleanup.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import av

from transcriber._exceptions import AudioFileError, ConversionError

logger = logging.getLogger(__name__)

# Formats supported directly by the Azure transcription API
API_SUPPORTED_FORMATS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}

# Formats that don't need conversion (mp4/webm always converted to strip video)
API_NO_CONVERSION = {".mp3", ".mpeg", ".mpga", ".m4a", ".wav"}

# Additional audio formats we can convert
AUDIO_CONVERTIBLE = {".aac", ".ogg", ".flac", ".wma", ".opus", ".aiff", ".aif"}

# Video formats we can extract audio from
VIDEO_CONVERTIBLE = {".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".3gp", ".mts", ".m2ts"}


def probe_audio_stream(file_path: Path) -> tuple[bool, str | None]:
    """Check whether a file contains an audio stream.

    Args:
        file_path: Path to the media file.

    Returns:
        ``(True, None)`` if an audio stream is found, otherwise
        ``(False, error_message)``.
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


def log_audio_file_info(file_path: Path) -> None:
    """Log information about the audio file format."""
    file_ext = file_path.suffix.lower()

    if file_ext in API_SUPPORTED_FORMATS:
        logger.debug("File format %s is directly supported by the API", file_ext)
    elif file_ext in AUDIO_CONVERTIBLE:
        logger.debug("Audio file %s will be converted to a supported format", file_ext)
    elif file_ext in VIDEO_CONVERTIBLE or file_ext == ".mp4":
        logger.debug("Video file %s detected — audio will be extracted", file_ext)
    else:
        logger.debug("File extension '%s' is not recognized", file_ext)
        logger.debug("Attempting to convert anyway...")
        logger.debug("API supports: %s", ", ".join(sorted(API_SUPPORTED_FORMATS)))


def get_audio_duration(file_path: str) -> float | None:
    """Get duration of an audio file in seconds using PyAV.

    Args:
        file_path: Path to the audio file.

    Returns:
        Duration in seconds, or ``None`` if it cannot be determined.
    """
    try:
        with av.open(file_path) as container:  # type: ignore[arg-type]
            if container.duration is not None:  # type: ignore[union-attr]
                return container.duration / 1_000_000  # type: ignore[union-attr]
            for stream in container.streams.audio:  # type: ignore[union-attr]
                if stream.duration is not None:
                    return float(stream.duration * stream.time_base)  # type: ignore[arg-type]
            return None
    except Exception:
        return None


def _convert_to_m4a(file_path: str) -> str:
    """Convert an audio/video file to M4A (AAC, mono, 16 kHz).

    Args:
        file_path: Source file path.

    Returns:
        Path to the converted temporary file.

    Raises:
        ConversionError: If conversion fails.
        AudioFileError: If the source file has no audio stream.
    """
    file_ext = Path(file_path).suffix.lower()

    if file_ext in {".mp4", ".webm"}:
        logger.debug(
            "%s file detected — extracting audio track to reduce file size",
            file_ext.upper(),
        )
    else:
        logger.debug("Converting %s to M4A format", file_ext)

    temp_file = tempfile.NamedTemporaryFile(suffix=".m4a", delete=False)
    temp_path = temp_file.name
    temp_file.close()

    try:
        input_container = av.open(file_path)  # type: ignore[arg-type]
        output_container = av.open(temp_path, mode="w")

        input_stream = None
        for stream in input_container.streams:
            if stream.type == "audio":
                input_stream = stream
                break

        if input_stream is None:
            input_container.close()
            output_container.close()
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise AudioFileError("No audio stream found in file", path=file_path)

        output_stream = output_container.add_stream("aac", rate=16000)  # type: ignore[union-attr]
        output_stream.bit_rate = 128000

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)

        for frame in input_container.decode(input_stream):  # type: ignore[arg-type]
            resampled_frames = resampler.resample(frame)  # type: ignore[arg-type]
            for resampled_frame in resampled_frames:
                for packet in output_stream.encode(resampled_frame):
                    output_container.mux(packet)

        for resampled_frame in resampler.resample(None):
            for packet in output_stream.encode(resampled_frame):
                output_container.mux(packet)

        for packet in output_stream.encode():
            output_container.mux(packet)

        input_container.close()
        output_container.close()
        logger.debug("Conversion complete")
        return temp_path

    except (AudioFileError, ConversionError):
        raise
    except av.error.FFmpegError as e:  # type: ignore[attr-defined]
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise ConversionError(f"Could not convert file: {e}", path=file_path) from e
    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise ConversionError(f"Unexpected error during conversion: {e}", path=file_path) from e


@contextmanager
def converted_audio(file_path: str) -> Generator[str, None, None]:
    """Context manager that yields a path to an API-compatible audio file.

    If the source format needs conversion, a temporary M4A file is created
    and automatically cleaned up on exit.

    Args:
        file_path: Path to the original audio/video file.

    Yields:
        Path to the (possibly converted) audio file.
    """
    file_ext = Path(file_path).suffix.lower()

    if file_ext in API_NO_CONVERSION:
        yield file_path
        return

    converted_path = _convert_to_m4a(file_path)
    try:
        yield converted_path
    finally:
        if os.path.exists(converted_path):
            logger.debug("Cleaning up converted audio file")
            os.unlink(converted_path)


def _split_audio_file(file_path: str, chunk_duration: int = 900) -> tuple[list[str], str]:
    """Split audio into chunks of the given duration.

    Args:
        file_path: Path to the audio file.
        chunk_duration: Maximum chunk length in seconds (default 900 = 15 min).

    Returns:
        Tuple of (list_of_chunk_paths, temp_directory).

    Raises:
        AudioFileError: If the file has no audio stream.
        ConversionError: If splitting fails.
    """
    logger.info("Splitting audio into %d-minute chunks", chunk_duration // 60)

    temp_dir = tempfile.mkdtemp()
    chunks: list[str] = []

    try:
        input_container = av.open(file_path)  # type: ignore[arg-type]

        audio_stream = None
        for stream in input_container.streams:
            if stream.type == "audio":
                audio_stream = stream
                break

        if audio_stream is None:
            input_container.close()
            raise AudioFileError("No audio stream found in file", path=file_path)

        chunk_index = 0
        chunk_start_time = 0
        output_container = None
        output_stream = None
        resampler = None

        for frame in input_container.decode(audio_stream):  # type: ignore[arg-type]
            frame_time = (
                float(frame.pts * frame.time_base)  # type: ignore[operator,arg-type,union-attr]
                if frame.pts is not None
                else 0
            )

            if frame_time >= chunk_start_time + chunk_duration:
                if output_container is not None and output_stream is not None:
                    if resampler is not None:
                        for resampled_frame in resampler.resample(None):
                            for packet in output_stream.encode(resampled_frame):
                                output_container.mux(packet)
                    for packet in output_stream.encode():
                        output_container.mux(packet)
                    output_container.close()
                    output_container = None
                    output_stream = None

                chunk_index += 1
                chunk_start_time = chunk_index * chunk_duration

            if output_container is None:
                chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index:03d}.m4a")
                chunks.append(chunk_path)
                output_container = av.open(chunk_path, mode="w")
                output_stream = output_container.add_stream("aac", rate=16000)  # type: ignore[assignment]
                output_stream.bit_rate = 128000
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)

            if resampler is not None and output_stream is not None:
                resampled_frames = resampler.resample(frame)  # type: ignore[arg-type]
                for resampled_frame in resampled_frames:
                    for packet in output_stream.encode(resampled_frame):
                        output_container.mux(packet)  # type: ignore[union-attr]

        if output_container is not None and output_stream is not None:
            if resampler is not None:
                for resampled_frame in resampler.resample(None):
                    for packet in output_stream.encode(resampled_frame):
                        output_container.mux(packet)
            for packet in output_stream.encode():
                output_container.mux(packet)
            output_container.close()

        input_container.close()

        logger.debug("Split into %d chunks", len(chunks))
        return chunks, temp_dir

    except (AudioFileError, ConversionError):
        raise
    except av.error.FFmpegError as e:  # type: ignore[attr-defined]
        raise ConversionError(f"Could not split file: {e}", path=file_path) from e
    except Exception as e:
        raise ConversionError(f"Unexpected error during split: {e}", path=file_path) from e


@contextmanager
def split_audio(file_path: str, chunk_duration: int = 900) -> Generator[list[str], None, None]:
    """Context manager that splits audio into chunks and cleans up afterwards.

    Args:
        file_path: Path to the audio file.
        chunk_duration: Maximum chunk length in seconds (default 900 = 15 min).

    Yields:
        List of paths to chunk files.
    """
    chunks, temp_dir = _split_audio_file(file_path, chunk_duration)
    try:
        yield chunks
    finally:
        logger.debug("Cleaning up temporary audio files")
        for chunk in chunks:
            if os.path.exists(chunk):
                os.unlink(chunk)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        logger.debug("Temporary files cleaned up")
