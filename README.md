# Audio Transcription Tool

A Python library and command-line tool for transcribing audio and video files to text.

> **Note:** Supports both **Azure OpenAI** (cloud) and **local Whisper** (offline) transcription.

## Features

- **Library + CLI** — Use as a Python package in your own code, or as a standalone CLI tool
- Supports almost any audio and video format you can think of (powered by ffmpeg)
- **Local transcription** — Use OpenAI Whisper for offline, private transcription (no API needed)
- Automatic speaker diarization and timestamps (Azure mode)
- **Glossary-based correction** — Use a custom glossary to fix industry terms, names, and acronyms
- **Synthesis** — Auto-generate a structured summary document from the transcript
- **Synthesise later** — Forgot `--synthesise`? Generate a synthesis from an existing transcript file
- **Parallel processing** — Long recordings are split into chunks and processed in parallel
- **Dependency injection** — Plug in custom transcription or LLM backends via Protocol types
- Simple command-line interface
- AI configured via environment variables

## Usage

```bash
transcribe <audio_file> [output_file] [options]
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--local` | `-l` | Off | Use local Whisper model instead of Azure API |
| `--model` | | `base` | Whisper model size (only with `--local`) |
| `--glossary` | `-g` | None | Path to a glossary file for transcript correction |
| `--synthesise` | `-s` | Off | Generate a synthesis/summary document (markdown) |
| `--synthesise-only` | `-S` | Off | Generate a synthesis from an existing transcript (skips transcription) |
| `--parallel-workers` | `-p` | 15 | Maximum parallel workers for chunk processing |

### Examples

```bash
# Azure transcription (requires API credentials)
transcribe meeting.mp3 meeting-transcript.txt

# Output file defaults to input filename with .txt extension
transcribe podcast-episode.m4a

# With glossary correction
transcribe meeting.mp3 transcript.txt --glossary company-terms.txt

# Long recording with custom parallelization
transcribe 2hour-webinar.m4a output.txt -g glossary.txt -p 10

# Local transcription (no API needed)
transcribe --local meeting.mp3

# Local with specific model
transcribe --local --model medium interview.wav

# Local transcription with cloud-based glossary correction
transcribe --local meeting.mp3 -g terms.txt

# Generate a synthesis/summary document
transcribe meeting.mp3 --synthesise

# Combine glossary correction and synthesis
transcribe meeting.mp3 --glossary terms.txt --synthesise
# (creates both meeting.txt and meeting_synthesis.md)

# Generate synthesis from an existing transcript
transcribe meeting.txt --synthesise-only
# (reads meeting.txt and creates meeting_synthesis.md)
```

## Installation

### Install via uv tool (recommended for CLI use)

```bash
# Cloud transcription only
uv tool install git+https://github.com/Deloitte-Nordics/transcriber.git

# With local Whisper support (~1.5 GB for models + PyTorch)
uv tool install "transcriber[local] @ git+https://github.com/Deloitte-Nordics/transcriber.git"
```

This installs `transcribe` as a global command available from anywhere.

### Install as a Python library

```bash
# Add to your project (cloud transcription only)
uv add "transcriber @ git+https://github.com/Deloitte-Nordics/transcriber.git"

# With local Whisper support
uv add "transcriber[local] @ git+https://github.com/Deloitte-Nordics/transcriber.git"
```

Or with pip:

```bash
pip install "transcriber @ git+https://github.com/Deloitte-Nordics/transcriber.git"
```

### Run without installing (uvx)

```bash
uvx --from git+https://github.com/Deloitte-Nordics/transcriber.git transcribe audio.mp3
```

## Library Usage

Use transcriber as a dependency in your own Python programs:

```python
import transcriber

# Simple cloud transcription (reads API credentials from env vars)
result = transcriber.transcribe_file("meeting.mp4")
print(result.transcript)

# With glossary correction and synthesis
result = transcriber.transcribe_file(
    "meeting.mp4",
    glossary="terms.txt",
    synthesise=True,
)
print(result.transcript)
print(result.synthesis)

# Write output to files automatically
result = transcriber.transcribe_file(
    "meeting.mp4",
    output="meeting.txt",
    synthesise=True,
)

# Local Whisper transcription (requires transcriber[local])
result = transcriber.transcribe_file("meeting.mp4", local=True, model="medium")

# Synthesise an existing transcript
synthesis = transcriber.synthesise_text("transcript text here...")
```

### Custom Backends (Dependency Injection)

Plug in your own transcription or LLM backends:

```python
import transcriber

# Use explicit Azure credentials (no env vars needed)
backend = transcriber.AzureTranscriptionBackend(
    api_key="your-key",
    api_url="https://your-endpoint.openai.azure.com/...",
)
result = transcriber.transcribe_file("meeting.mp4", transcription_backend=backend)

# Or implement the Protocol for a custom provider
class MyTranscriptionBackend:
    def transcribe(self, audio_path: str, *, time_offset: int = 0) -> str:
        # Your custom transcription logic
        ...

result = transcriber.transcribe_file("meeting.mp4", transcription_backend=MyTranscriptionBackend())
```

### Return Types

`transcribe_file()` returns a `TranscriptionResult`:

```python
@dataclass(frozen=True)
class TranscriptionResult:
    transcript: str              # The transcription text
    synthesis: str | None        # Synthesis markdown, or None
    duration_seconds: float | None  # Audio duration in seconds, or None
```

### Exception Handling

All errors are typed exceptions:

```python
import transcriber

try:
    result = transcriber.transcribe_file("meeting.mp4")
except transcriber.ConfigurationError as e:
    print(f"Config issue: {e.errors}")
except transcriber.AudioFileError as e:
    print(f"File issue: {e} (path: {e.path})")
except transcriber.TranscriptionError as e:
    print(f"API issue: {e}")
except transcriber.TranscriberError as e:
    print(f"General error: {e}")
```

## Configuration

This tool requires Azure OpenAI Service with two model deployments:

### Required Model Deployments

1. **Transcription model:** Deploy `gpt-4o-transcribe` (or newer) for audio transcription with speaker diarization.

2. **Chat completion model** (for glossary correction and synthesis): Deploy any GPT-4 class model or better (e.g., `gpt-4o`, `gpt-4o-mini`). A smaller model like `gpt-4o-mini` is sufficient for this task.

### Environment Variables

Add the following environment variables to your `~/.zshrc` file:

```bash
# Required for transcription
export AZURE_TRANSCRIBE_API_KEY="your-api-key-here"
export AZURE_TRANSCRIBE_URL="https://your-endpoint.openai.azure.com/openai/deployments/<your-transcribe-deployment>/audio/transcriptions?api-version=2025-03-01-preview"

# Required only when using --glossary, --synthesise, or --synthesise-only
export AZURE_TEXT_API_KEY="your-text-api-key-here"
export AZURE_TEXT_URL="https://your-endpoint.openai.azure.com/openai/deployments/<your-chat-deployment>/chat/completions?api-version=2025-03-01-preview"
```

Replace:
- `your-endpoint` with your Azure OpenAI endpoint (found in **Keys and Endpoint**)
- `<your-transcribe-deployment>` with your transcription model deployment name
- `<your-chat-deployment>` with your chat completion model deployment name

Then reload your shell configuration:

```bash
source ~/.zshrc
```

## Local Transcription Mode

> **Requires the `[local]` extra:** Install with `uv tool install "transcriber[local] @ ..."` or `uv add "transcriber[local] @ ..."`.

Use the `--local` flag to transcribe audio offline using OpenAI's Whisper model. No API credentials are needed for transcription (though glossary correction and synthesis still require Azure LLM credentials).

### Available Models

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `tiny` | 39 MB | Fastest | Quick drafts, testing |
| `base` | 74 MB | Fast | Default — good balance of speed and accuracy |
| `small` | 244 MB | Medium | Better accuracy for clear audio |
| `medium` | 769 MB | Slow | High accuracy, multiple languages |
| `large-v3` | 1.55 GB | Slowest | Best accuracy, all languages |
| `turbo` | 809 MB | Fast | Near-large accuracy at much higher speed |

> **Tip:** English-only models (`tiny.en`, `base.en`, `small.en`, `medium.en`) are faster and more accurate for English-only audio.

> **Note:** Models are downloaded automatically on first use and cached locally (~/.cache/whisper/).

### Limitations

- **No speaker diarization** — Local mode does not identify speakers (output shows timestamps only)
- **GPU recommended** — Large models benefit from CUDA GPU; CPU works but is slower
- **First run downloads model** — Initial use of a model downloads it (may take a few minutes)

## Glossary-Based Correction

When you provide a glossary file with `--glossary`, the tool will:

1. Transcribe the audio using speech-to-text
2. Send the transcript to an LLM along with your glossary
3. The LLM corrects likely mishearings based on the glossary terms

The glossary file can be **any text format** — the LLM will understand it. Examples:

**Simple list:**
```
ACME Corporation
John Smith
Q4 2025
API Gateway
```

**With context:**
```
Company Names:
- ACME Corporation (our main client)
- TechFlow Inc.

People:
- John Smith (CEO)
- Sarah Johnson (CTO)

OR

| Term | Full Name | Description |
|------|-----------|-------------|
| **HCP** | Hearing Care Professional | Audiologist, dispenser, or fitter |
| **HAW** | Hearing Aid Wearer | End-user/patient/consumer |
```

**Or any other format** — JSON, Markdown tables, prose descriptions, etc.

## Parallel Processing

For recordings longer than ~23 minutes, the tool automatically:

1. Splits the audio into 15-minute chunks
2. Processes up to 15 chunks in parallel (configurable with `--parallel-workers`)
3. Each chunk is transcribed and (if glossary provided) corrected independently
4. Results are combined in the correct order

This significantly speeds up processing of long recordings.

## Synthesis

Use `--synthesise` (or `-s`) to automatically generate a structured summary of the transcript:

```bash
transcribe meeting.mp3 --synthesise
transcribe --local meeting.mp3 -s
transcribe meeting.mp3 --glossary terms.txt --synthesise
```

This creates an additional markdown file alongside the transcript (e.g., `meeting_synthesis.md`) containing a structured summary of the conversation.

### Synthesise an existing transcript

If you already have a transcript and want to generate a synthesis after the fact (e.g., you forgot to pass `--synthesise` during transcription), use `--synthesise-only` (or `-S`):

```bash
transcribe meeting.txt --synthesise-only
transcribe meeting.txt -S
```

This reads the transcript file and creates `meeting_synthesis.md` — no audio processing or transcription credentials are needed.

> **Note:** Synthesis requires Azure LLM credentials (`AZURE_TEXT_API_KEY` and `AZURE_TEXT_URL`), even when using `--local` for transcription or `--synthesise-only`.

## Output Format

The tool saves transcriptions as plain text files, including:
- Speaker diarization (speaker labels, Azure mode only)
- Timestamps
- Complete transcription text

Example output (Azure mode with speaker diarization):
```
[0.00s - 5.23s] Speaker 1: Welcome to today's meeting.
[5.45s - 12.10s] Speaker 2: Thanks, let's start with the Q4 review.
```

### Local mode output

When using `--local`, output includes timestamps but not speaker labels:
- Timestamps
- Complete transcription text

Note: Speaker diarization is only available with Azure OpenAI transcription.

## Requirements

- Python 3.10 or higher
- Azure OpenAI Service with:
  - A transcription model deployment (e.g., gpt-4o-transcribe) — not needed for `--local` or `--synthesise-only`
  - A chat completion model deployment (for `--glossary`, `--synthesise`, and `--synthesise-only`)
- **Optional:** `openai-whisper` for local transcription — install with the `[local]` extra
