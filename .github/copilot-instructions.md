# Project Guidelines

## Code Style
- Python 3.10+, strict type hints everywhere (Pyright strict mode, `from __future__ import annotations`)
- Line length: 100 chars, enforced by Ruff
- Google-style docstrings with Args/Returns/Raises sections
- Internal modules use underscore prefix (`_audio.py`); public API exposed via `__all__` in [transcriber/\_\_init\_\_.py](transcriber/__init__.py)
- Frozen dataclasses for immutable results; `@runtime_checkable` Protocols for DI
- Reference patterns: [_protocols.py](transcriber/_protocols.py) (interfaces), [_audio.py](transcriber/_audio.py) (context managers), [_exceptions.py](transcriber/_exceptions.py) (error hierarchy)

## Architecture
- **Public API**: `transcribe_file()` and `synthesise_text()` in [transcriber/\_\_init\_\_.py](transcriber/__init__.py) — all other modules are internal
- **Protocol-based DI**: Backends implement `TranscriptionBackend` or `LLMBackend` from [_protocols.py](transcriber/_protocols.py) — never hardcode a backend
- **Separation rule**: Library code raises typed exceptions inheriting `TranscriberError`; only [cli.py](transcriber/cli.py) calls `sys.exit()`
- **Config validation**: Accumulate ALL errors before raising ([_config.py](transcriber/_config.py))
- **Prompt templates**: Markdown files loaded via `importlib.resources` with `{{placeholder}}` syntax ([correction_prompt.md](transcriber/correction_prompt.md), [synthesis_prompt.md](transcriber/synthesis_prompt.md))
- **Parallel chunks**: Files >23m20s split into 15-min chunks, processed via `ThreadPoolExecutor`

## Build & Test
```bash
make install-dev   # Install with dev dependencies
make lint          # Ruff + Pyright (run before committing)
make fix           # Auto-fix style issues
make test          # Full test suite (needs Azure creds)
make test-unit     # Unit tests only (no API calls)
make test-fast     # Tests that skip audio/API
make release       # Auto-detect bump from commits, tag, push
```
- Follow TDD: failing test → minimal code → passing test
- Unit tests use `MagicMock` backends ([test_api.py](tests/test_api.py)); integration tests use real Azure API
- Session-scoped fixtures for expensive resources; skip gracefully when creds missing ([conftest.py](tests/conftest.py))
- Negative/error-handling tests isolated in [test_negative.py](tests/test_negative.py)

## Commit Format (REQUIRED)
Conventional Commits for automated versioning via [git-cliff](cliff.toml):
- `fix:` → patch (0.0.X) | `feat:` → minor (0.X.0) | `feat!:` / `fix!:` / `BREAKING CHANGE:` → major (X.0.0)
- `chore:`, `docs:`, `test:`, `refactor:`, `style:` → no version bump

## Integration Points
- **Azure OpenAI**: Transcription (`AZURE_TRANSCRIBE_API_KEY`, `AZURE_TRANSCRIBE_URL`) and LLM (`AZURE_TEXT_API_KEY`, `AZURE_TEXT_URL`)
- **Local Whisper**: Lazy-imported only with `--local` flag; optional `[local]` extra
- **FFmpeg** (via PyAV): Format conversion and audio splitting in [_audio.py](transcriber/_audio.py) — always use context managers for temp files
- **Retry**: Exponential backoff (2s/4s/8s) in [_retry.py](transcriber/_retry.py); glossary correction falls back to uncorrected on failure
