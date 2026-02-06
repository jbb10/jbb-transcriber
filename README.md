# Audio Transcription CLI Tool

A command-line tool for transcribing audio and video files to text.

> **Note:** Currently, only **Azure OpenAI** is supported. Other AI providers may be added in the future.

## Features

- Transcribe audio files in various formats (MP3, M4A, WAV, AAC, etc.)
- Automatic speaker diarization and timestamps
- **Glossary-based correction** — Use a custom glossary to fix industry terms, names, and acronyms
- **Parallel processing** — Long recordings are split into chunks and processed in parallel
- Simple command-line interface
- AI configured via environment variables

## Installation

### Install via uv tool (recommended)

```bash
uv tool install git+https://github.com/Deloitte-Nordics/transcriber.git
```

This installs `transcribe` as a global command available from anywhere.

### Run without installing (uvx)

```bash
uvx --from git+https://github.com/Deloitte-Nordics/transcriber.git transcribe audio.mp3
```

### Install into a virtual environment

If you prefer to install into a specific environment:

```bash
uv pip install git+https://github.com/Deloitte-Nordics/transcriber.git
# or
pip install git+https://github.com/Deloitte-Nordics/transcriber.git
```

## Configuration

This tool requires Azure OpenAI Service with two model deployments:

### Required Model Deployments

1. **Transcription model:** Deploy `gpt-4o-transcribe` (or newer) for audio transcription with speaker diarization.

2. **Chat completion model** (only for glossary correction): Deploy any GPT-4 class model or better (e.g., `gpt-4o`, `gpt-4o-mini`). A smaller model like `gpt-4o-mini` is sufficient for this task.

### Environment Variables

Add the following environment variables to your `~/.zshrc` file:

```bash
# Required for transcription
export AZURE_TRANSCRIBE_API_KEY="your-api-key-here"
export AZURE_TRANSCRIBE_URL="https://your-endpoint.openai.azure.com/openai/deployments/<your-transcribe-deployment>/audio/transcriptions?api-version=2025-03-01-preview"

# Required only when using --glossary for transcript correction
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

## Usage

```bash
transcribe <audio_file> [output_file] [options]
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--glossary` | `-g` | None | Path to a glossary file for transcript correction |
| `--parallel-workers` | `-p` | 15 | Maximum parallel workers for chunk processing |

### Examples

```bash
# Basic transcription
transcribe meeting.mp3 meeting-transcript.txt

# Output file defaults to input filename with .txt extension
transcribe podcast-episode.m4a

# With glossary correction
transcribe meeting.mp3 transcript.txt --glossary company-terms.txt

# Long recording with custom parallelization
transcribe 2hour-webinar.m4a output.txt -g glossary.txt -p 10
```

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

Technical Terms:
- Kubernetes (often misheard as "Cooper Netties")
- OAuth (not "Oh Auth")
```

**Or any other format** — JSON, Markdown tables, prose descriptions, etc.

## Parallel Processing

For recordings longer than ~23 minutes, the tool automatically:

1. Splits the audio into 15-minute chunks
2. Processes up to 15 chunks in parallel (configurable with `--parallel-workers`)
3. Each chunk is transcribed and (if glossary provided) corrected independently
4. Results are combined in the correct order

This significantly speeds up processing of long recordings.

## Supported Audio Formats

- MP3 (.mp3)
- MP4 Audio (.m4a, .mp4)
- WAV (.wav)
- AAC (.aac)
- OGG (.ogg)
- FLAC (.flac)
- WebM (.webm)
- And more...

## Output Format

The tool saves transcriptions as plain text files, including:
- Speaker diarization (speaker labels)
- Timestamps
- Complete transcription text

Example output:
```
[0.00s - 5.23s] Speaker 1: Welcome to today's meeting.
[5.45s - 12.10s] Speaker 2: Thanks, let's start with the Q4 review.
```

## Error Handling

The tool will display clear error messages for:
- Missing environment variables
- Audio file not found
- Unsupported file formats
- API request failures

For glossary correction, the tool includes retry logic:
- 3 attempts with exponential backoff (2s, 4s, 8s)
- Falls back to uncorrected transcript if all retries fail

## Uninstallation

```bash
# If installed via uv tool
uv tool uninstall transcriber

# If installed via pip/uv pip
uv pip uninstall transcriber
# or
pip uninstall transcriber
```

## Versioning

This project uses [Semantic Versioning](https://semver.org/). Versions are tracked via git tags.

### To release a new version:

1. Update version in `pyproject.toml` and `transcriber.py`
2. Commit the changes: `git commit -am "Bump version to X.Y.Z"`
3. Create a git tag: `git tag vX.Y.Z`
4. Push with tags: `git push && git push --tags`

Users can then install a specific version:

```bash
uv tool install git+https://github.com/Deloitte-Nordics/transcriber.git@vX.Y.Z
```

## Requirements

- Python 3.10 or higher
- Azure OpenAI Service with:
  - A transcription model deployment (e.g., gpt-4o-transcribe)
  - A chat completion model deployment (only for glossary feature)
