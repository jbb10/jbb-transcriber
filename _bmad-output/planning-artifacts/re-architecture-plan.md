# Plan: Transcriber Re-architecture with Async, DI & Multi-Provider Support

## TL;DR

Re-architect the transcriber from a sync, Azure-coupled codebase into an async-first, provider-agnostic library with clean dependency injection. The 370-line `__init__.py` god-module gets decomposed into a thin re-export surface + dedicated pipeline, backend, and configuration modules. `httpx.AsyncClient` replaces `requests`. A `backends/` package isolates provider-specific code behind stable protocols, making Anthropic (or any provider) a one-module addition. All 14 adversarial review findings are resolved structurally.

---

## Phase 1: Foundation — Types, Exceptions, Protocols (no dependencies between steps)

### Step 1.1: Result types → `_types.py`
- Extract `ChunkResult` and `TranscriptionResult` from `__init__.py` into new `transcriber/_types.py`
- No logic changes — just relocation

### Step 1.2: Exception hierarchy → `_exceptions.py` (unchanged location, minor additions)
- Add `PromptError` for template issues
- Add `SecurityError(TranscriberError)` for URL scheme violations
- Keep existing hierarchy intact

### Step 1.3: Protocols → `_protocols.py` (make async)
- Change `TranscriptionBackend.transcribe()` → `async def transcribe()`
- Change `LLMBackend.complete()` → `async def complete()`
- Remove `max_retries` from `LLMBackend.complete()` signature — retry is a cross-cutting concern, not a backend responsibility. Backends do one call; the pipeline layer handles retries.
- Keep `@runtime_checkable`

**New protocol signatures:**
```python
class TranscriptionBackend(Protocol):
    async def transcribe(self, audio_path: str, *, time_offset: int = 0) -> str: ...

class LLMBackend(Protocol):
    async def complete(self, prompt: str, *, temperature: float = 0.1) -> str: ...
```

### Step 1.4: Settings dataclasses → `_settings.py` (new)
- Plain `dataclasses` (stdlib, no pydantic dependency for the library)
- Grouped settings replace scattered constructor params and hard-coded defaults
- Each provider has its own settings dataclass
- Factory classmethods `from_env()` on each settings class handle env var loading

```python
@dataclass(frozen=True)
class AzureTranscriptionSettings:
    api_key: str
    api_url: str
    model: str = "gpt-4o-transcribe-diarize"
    request_timeout: int = 600

    @classmethod
    def from_env(cls) -> AzureTranscriptionSettings: ...

@dataclass(frozen=True)
class AzureLLMSettings:
    api_key: str
    api_url: str
    model: str = "gpt-5.1"
    request_timeout: int = 300

    @classmethod
    def from_env(cls) -> AzureLLMSettings: ...

@dataclass(frozen=True)
class WhisperSettings:
    model_name: str = "base"
    device: str | None = None  # auto-detect

@dataclass(frozen=True)
class PipelineSettings:
    chunk_duration: int = 900         # seconds per chunk
    parallel_workers: int = 15
    max_duration_before_split: int = 1400  # seconds
    max_retries: int = 3              # retry attempts per backend call
    base_delay: float = 2.0           # base backoff delay in seconds
    max_glossary_size: int = 500_000  # bytes — input size safety limit
    fail_on_correction_error: bool = False  # if True, raise on glossary correction failure
```

**Retry config ownership:** `max_retries` and `base_delay` live **only** on `PipelineSettings` — they are pipeline-level concerns, not backend settings. Backend settings dataclasses (`AzureTranscriptionSettings`, `AzureLLMSettings`) do NOT contain retry parameters. The pipeline wraps each backend call in `retry_with_backoff()`, passing these values.

**Rationale for stdlib dataclasses over pydantic-settings:**
- Library users shouldn't inherit a pydantic dependency just to use `transcribe_file()`
- Constructor injection is explicit — callers build settings, pass them in
- `from_env()` classmethods provide convenience without framework coupling
- Validation lives in `from_env()` and in the pipeline, not in the settings type itself

---

## Phase 2: Backends Package — Provider Isolation (*parallel with Phase 1*)

### Step 2.1: Create `transcriber/backends/` package
- `transcriber/backends/__init__.py` — re-exports all backends + factory functions
- `transcriber/backends/_azure.py` — `AzureTranscriptionBackend`, `AzureLLMBackend`
- `transcriber/backends/_whisper.py` — `WhisperTranscriptionBackend`

### Step 2.2: Async Azure backends (`backends/_azure.py`)
- Replace `requests` with `httpx.AsyncClient`
- Accept settings dataclass in constructor (not individual params)
- **URL scheme validation** in both `from_env()` (earliest boundary) and constructor (defense in depth) → fixes adversarial finding #1
- Model name comes from settings, not hard-coded → fixes finding #8
- Response parsing extracted into private methods for clarity
- No `from_env()` on backends — that's on the settings class
- Backend implements `AsyncContextManager` protocol for lifecycle management
- Backend owns its own `httpx.AsyncClient` lifecycle (create in constructor or accept injected)

```python
class AzureTranscriptionBackend:
    def __init__(self, settings: AzureTranscriptionSettings, *, client: httpx.AsyncClient | None = None):
        validate_https_url(settings.api_url)
        self._settings = settings
        self._client = client or httpx.AsyncClient(timeout=settings.request_timeout)
        self._owns_client = client is None

    async def transcribe(self, audio_path: str, *, time_offset: int = 0) -> str: ...
    async def aclose(self) -> None: ...  # cleanup client if owned

    async def __aenter__(self) -> AzureTranscriptionBackend: return self
    async def __aexit__(self, *exc: object) -> None: await self.aclose()
```

**Note:** `validate_https_url()` is called in `from_env()` on the settings class (earliest boundary — catches bad env vars immediately) AND in the backend constructor (defense in depth — catches programmatic misuse). The settings-level check gives a better error message referencing the env var name.

### Step 2.3: Whisper backend (`backends/_whisper.py`)
- Keep sync internally (whisper is CPU-bound), wrap in `asyncio.to_thread()`
- Accept `WhisperSettings` dataclass

```python
class WhisperTranscriptionBackend:
    def __init__(self, settings: WhisperSettings):
        self._settings = settings

    async def transcribe(self, audio_path: str, *, time_offset: int = 0) -> str:
        return await asyncio.to_thread(self._transcribe_sync, audio_path, time_offset)
```

### Step 2.4: Backend factory (`backends/__init__.py`)
- `create_transcription_backend(provider: str, **kwargs)` — dispatches to the right backend
- `create_llm_backend(provider: str, **kwargs)` — dispatches to the right backend
- Makes adding Anthropic a matter of: (a) add `backends/_anthropic.py`, (b) register in factory

---

## Phase 3: Infrastructure — Retry, Prompts, Audio, Security

### Step 3.1: Async retry → `_retry.py`
- Convert `retry_with_backoff` to `async def retry_with_backoff()`
- Use `asyncio.sleep()` instead of `time.sleep()`
- `is_transient_http_error()` updated for `httpx` exceptions (`httpx.HTTPStatusError`, `httpx.TimeoutException`, etc.)
- Keep the same backoff logic (exponential + jitter + Retry-After)

### Step 3.2: Prompts → `_prompts.py` (new, extracted from `_llm.py`)
- Move `_load_prompt()`, `build_correction_prompt()`, `build_synthesis_prompt()` here
- **Fix naive `str.replace` template injection** → use single-pass replacement to prevent double-substitution → fixes adversarial finding #2. Approach: replace all markers in one pass using `re.sub` with a dict lookup, so user content that happens to contain `{{glossary}}` or `{{transcript}}` is never interpreted as a marker.

```python
def _render_template(template: str, substitutions: dict[str, str]) -> str:
    """Replace template markers in a single pass (prevents double-substitution)."""
    import re
    pattern = re.compile("|".join(re.escape(k) for k in substitutions))
    return pattern.sub(lambda m: substitutions[m.group(0)], template)

def build_correction_prompt(transcript: str, glossary_text: str) -> str:
    template = _load_prompt("correction_prompt.md")
    return _render_template(template, {"{{glossary}}": glossary_text, "{{transcript}}": transcript})
```

**Rationale:** Rejecting user content that contains marker strings is too restrictive — a software project glossary might legitimately contain `{{glossary}}`. Single-pass replacement eliminates the vulnerability without restricting input. Note: `_load_prompt` relocation from `_llm.py` to `_prompts.py` is transparent to packaging — `pkg_files("transcriber")` works identically from any module inside the package.

### Step 3.3: Audio processing → `_audio.py` (cleanup)
- Fix temp file leak in `_convert_to_m4a` — use `try/finally` from the moment `NamedTemporaryFile` is created → fixes finding #3
- Fix temp dir leak in `_split_audio_file` — wrap entire processing loop in `try/finally` with `shutil.rmtree(temp_dir)` so the temp dir is cleaned up even if an exception is raised mid-processing (before the context manager's `finally` block gets a chance to run) → fixes finding #4
- Fix `split_audio` cleanup — replace individual `os.unlink` + `os.rmdir` with a single `shutil.rmtree(temp_dir)` call (rmtree handles all contents) → fixes finding #12
- Audio processing stays **sync** — PyAV is not async. The pipeline wraps calls in `asyncio.to_thread()`.

```python
# _split_audio_file — fixed cleanup
def _split_audio_file(file_path: str, chunk_duration: int = 900) -> tuple[list[str], str]:
    temp_dir = tempfile.mkdtemp()
    try:
        # ... all processing ...
        return chunks, temp_dir
    except (AudioFileError, ConversionError):
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise ConversionError(...) from e

# split_audio context manager — simplified cleanup
@contextmanager
def split_audio(file_path: str, chunk_duration: int = 900):
    chunks, temp_dir = _split_audio_file(file_path, chunk_duration)
    try:
        yield chunks
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
```

### Step 3.4: Security validation → `_security.py` (new)
- `validate_https_url(url: str) → None` — raises `SecurityError` if scheme is not `https`
- `validate_input_size(content: str, max_bytes: int, name: str) → None` — raises `ConfigurationError` if too large → fixes finding #7
- Used by settings `from_env()` (URL validation at earliest boundary), backends (URL validation as defense in depth), and pipeline (glossary size validation)

---

## Phase 4: Pipeline — Orchestration (*depends on Phases 1–3*)

### Step 4.1: `_pipeline.py` (new — core orchestration extracted from `__init__.py`)
- All business logic currently in `__init__.py::transcribe_file()` moves here
- Fully async

```python
async def transcribe(
    path: str | os.PathLike[str],
    *,
    output: str | os.PathLike[str] | None = None,
    glossary: str | os.PathLike[str] | None = None,
    synthesise: bool = False,
    transcription_backend: TranscriptionBackend,   # required — no implicit instantiation
    llm_backend: LLMBackend | None = None,         # required if glossary or synthesise
    settings: PipelineSettings = PipelineSettings(),
    on_chunk_complete: Callable[[int, int], None] | None = None,
) -> TranscriptionResult: ...

async def synthesise(
    transcript: str,
    *,
    llm_backend: LLMBackend,
) -> str: ...
```

**Key design decision:** Backends are **required** constructor params in the pipeline — no implicit `from_env()` inside the library. The library never reads env vars on its own. Env var resolution is the caller's responsibility (CLI does it; library users do it themselves or use `Settings.from_env()`). This eliminates dual validation paths and makes DI explicit.

### Step 4.2: Parallel chunk processing (async)
- Replace `ThreadPoolExecutor` with `asyncio.Semaphore` + `asyncio.gather()`
- Semaphore limits concurrency → fixes adversarial finding #5 (thundering herd)

```python
async def _transcribe_chunks(
    chunks: list[tuple[int, str, int]],
    backend: TranscriptionBackend,
    llm: LLMBackend | None,
    glossary_text: str | None,
    settings: PipelineSettings,
    on_chunk_complete: Callable[[int, int], None] | None,
) -> list[ChunkResult]:
    semaphore = asyncio.Semaphore(settings.parallel_workers)

    async def _process(info: tuple[int, str, int]) -> ChunkResult:
        async with semaphore:
            return await _transcribe_single_chunk(info, backend, llm, glossary_text, settings)

    return await asyncio.gather(*[_process(info) for info in chunks])
```

**Retry placement:** The pipeline wraps each backend call in `retry_with_backoff()`. This happens inside `_transcribe_single_chunk`:

```python
async def _transcribe_single_chunk(
    chunk_info: tuple[int, str, int],
    backend: TranscriptionBackend,
    llm: LLMBackend | None,
    glossary_text: str | None,
    settings: PipelineSettings,
) -> ChunkResult:
    index, chunk_path, time_offset = chunk_info
    # Pipeline owns retry — backend does a single call
    text = await retry_with_backoff(
        lambda: backend.transcribe(chunk_path, time_offset=time_offset),
        max_retries=settings.max_retries,
        base_delay=settings.base_delay,
        operation_name=f"transcription chunk {index}",
    )
    if glossary_text and llm is not None:
        correction = await _correct_with_glossary(text, glossary_text, llm, settings)
        text = correction.text
    return ChunkResult(index=index, transcript=text, ...)
```

Similarly, `_correct_with_glossary` and `synthesise` wrap the `llm.complete()` call in `retry_with_backoff()`. Backends never retry internally.

### Step 4.3: Glossary correction handling
- `correct_with_glossary()` → returns a `CorrectionResult` dataclass instead of silently falling back → fixes finding #10
- Pipeline decides policy: log warning and continue, or raise — configurable via `PipelineSettings.fail_on_correction_error`

```python
@dataclass(frozen=True)
class CorrectionResult:
    text: str
    was_corrected: bool

async def _correct_with_glossary(
    transcript: str,
    glossary_text: str,
    llm: LLMBackend,
    settings: PipelineSettings,
) -> CorrectionResult:
    prompt = build_correction_prompt(transcript, glossary_text)
    try:
        corrected = await retry_with_backoff(
            lambda: llm.complete(prompt, temperature=0.1),
            max_retries=settings.max_retries,
            base_delay=settings.base_delay,
            operation_name="glossary correction",
        )
        return CorrectionResult(text=corrected, was_corrected=True)
    except (LLMError, SynthesisError):
        if settings.fail_on_correction_error:
            raise
        logger.warning("Glossary correction failed — using uncorrected text")
        return CorrectionResult(text=transcript, was_corrected=False)
```

### Step 4.4: Input validation in pipeline
- Validate glossary file size before reading → fixes finding #7
- Validate `duration is not None` before chunking decision; if `None`, log warning and attempt single-file transcription → fixes finding #14

---

## Phase 5: Public API Surface — `__init__.py` (*depends on Phase 4*)

### Step 5.1: Thin `__init__.py`
- **Only re-exports** — no business logic
- Provide sync convenience wrappers that call `asyncio.run()` on the pipeline

```python
def _run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an async coroutine synchronously. Raises RuntimeError with a clear
    message if called from within an existing event loop (e.g. Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass  # No loop running — safe to use asyncio.run()
    else:
        raise RuntimeError(
            "transcribe_file() cannot be called from an async context. "
            "Use 'await transcriber.transcribe(...)' instead."
        )
    return asyncio.run(coro)

# Sync wrapper (convenience for library users)
def transcribe_file(
    path: str | os.PathLike[str],
    *,
    transcription_backend: TranscriptionBackend,
    llm_backend: LLMBackend | None = None,
    settings: PipelineSettings = PipelineSettings(),
    **kwargs,
) -> TranscriptionResult:
    return _run_sync(pipeline.transcribe(path, transcription_backend=transcription_backend, llm_backend=llm_backend, settings=settings, **kwargs))

# Async API (primary)
transcribe = pipeline.transcribe
synthesise = pipeline.synthesise
```

### Step 5.2: `__all__` updated
- Export result types, exceptions, protocols, backends, settings, factory functions
- Export both sync (`transcribe_file`) and async (`transcribe`) entry points

---

## Phase 6: CLI (*depends on Phase 5*)

### Step 6.1: `cli.py` cleanup
- **Remove `sys.exit()` from `_run()`** — replace with typed exceptions → fixes finding #9
  - Empty transcript file → `ConfigurationError(["Transcript file is empty"])`
  - Missing LLM backend → `ConfigurationError(["LLM backend required for synthesis"])`
- `_run()` becomes `async def _run()`, `main()` calls `asyncio.run()`
- CLI is the **only place** that reads env vars (via `Settings.from_env()`) and constructs backends
- CLI validates inputs, builds settings + backends, calls `transcriber.transcribe()`
- CLI manages backend lifecycle via `async with` (backends implement `AsyncContextManager`):

```python
async def _run(validated: ValidatedConfig) -> None:
    settings = PipelineSettings(...)
    async with AzureTranscriptionBackend(transcription_settings) as t_backend:
        llm_backend = AzureLLMBackend(llm_settings) if need_llm else None
        try:
            await transcriber.transcribe(..., transcription_backend=t_backend, llm_backend=llm_backend)
        finally:
            if llm_backend is not None:
                await llm_backend.aclose()
```

### Step 6.2: Provider selection (prep for multi-provider)
- Add `--provider` flag (default: `"azure"`, choices: `["azure"]` — Anthropic added later)
- CLI dispatches to the right settings class and backend factory based on provider
- Error messages reference provider-specific env var names

### Step 6.3: `ValidatedConfig` refactored
- Remove raw credential strings from `ValidatedConfig` — it should hold validated paths and flags
- Backends constructed by `_run()` from settings, not from `ValidatedConfig` credential fields
- This means `ValidatedConfig` no longer knows about Azure-specific env vars

---

## Phase 7: Dependency Changes

### Step 7.1: Replace `requests` with `httpx`
- `pyproject.toml`: remove `requests>=2.31.0`, add `httpx>=0.27.0`
- **Both sync and async** support in httpx — sync wrapper still works
- Update retry logic for httpx exception types

### Step 7.2: Update `pyproject.toml`
```toml
dependencies = [
    "httpx>=0.27.0",
    "av>=12.0.0",
]
```

---

## Phase 8: Tests (*depends on all above*)

### Step 8.1: Test infrastructure
- Add async test support: `pytest-asyncio` to dev dependencies
- Update `conftest.py` fixtures to be provider-neutral (accept backend protocol, not Azure-specific fixture)
- Integration tests still use Azure (they need real API) but fixture names become generic

### Step 8.2: New unit tests
- `test_security.py` — URL scheme validation, input size limits
- `test_pipeline.py` — async pipeline with mock backends (replaces some of `test_api.py`)
- `test_audio_cleanup.py` — temp file leak scenarios → fixes finding #13
- `test_prompts.py` — template marker validation

### Step 8.3: Existing test migration
- Tests import from new module paths
- Mock backends implement async protocol
- `test_retry.py` updated for async + httpx exceptions

---

## Relevant Files

### New files to create
- `transcriber/_types.py` — result dataclasses extracted from `__init__.py`
- `transcriber/_settings.py` — settings dataclasses with `from_env()` classmethods
- `transcriber/_pipeline.py` — async orchestration logic extracted from `__init__.py`
- `transcriber/_prompts.py` — prompt loading extracted from `_llm.py`
- `transcriber/_security.py` — URL and input size validation
- `transcriber/backends/__init__.py` — factory functions + re-exports
- `transcriber/backends/_azure.py` — Azure backends extracted from `_llm.py` and `_transcription.py`
- `transcriber/backends/_whisper.py` — Whisper backend extracted from `_transcription.py`

### Files to heavily modify
- `transcriber/__init__.py` — gut to thin re-export surface + sync wrappers (~50 lines, down from 370+)
- `transcriber/_retry.py` — convert to async, update exception types for httpx
- `transcriber/_audio.py` — fix cleanup bugs (findings #3, #4, #12)
- `transcriber/cli.py` — remove `sys.exit()` from `_run()`, add `--provider` flag, make `_run()` async
- `transcriber/_protocols.py` — make methods async, remove `max_retries` from `LLMBackend`
- `transcriber/_exceptions.py` — add `PromptError`, `SecurityError`
- `pyproject.toml` — swap `requests` → `httpx`, add `pytest-asyncio`

### Files to delete
- `transcriber/_llm.py` — split into `backends/_azure.py` (backend class) + `_prompts.py` (prompt helpers)
- `transcriber/_transcription.py` — split into `backends/_azure.py` + `backends/_whisper.py`
- `transcriber/_config.py` — shim no longer needed (was just re-exporting from cli.py)

### Import migration for `_config.py` deletion
The following files import from `transcriber._config` and must be updated to import from `transcriber.cli` instead:
- `tests/test_cli.py` — imports `ValidatedConfig`, `validate_cli_config`
- Any other test files referencing `_config` (search with `grep -r '_config' tests/`)

### Files mostly unchanged
- `transcriber/_audio.py` — bug fixes only, stays sync
- `transcriber/correction_prompt.md` — no change
- `transcriber/synthesis_prompt.md` — no change
- `transcriber/__main__.py` — no change

---

## Verification

1. **All existing unit tests pass** after migration (with import path updates)
2. **`pyright --strict`** passes with no new errors
3. **`ruff check`** passes
4. **`pytest tests/test_api.py tests/test_negative.py tests/test_retry.py`** — these are mock-based and must pass without credentials
5. **Manual CLI test**: `transcribe tests/fixtures/short_speech.mp3` works end-to-end (requires Azure creds)
6. **Manual async test**: `await transcriber.transcribe(...)` works from an async context
7. **Security check**: Confirm `http://` URLs are rejected by backends
8. **Temp file cleanup check**: Kill process mid-transcription, verify no orphaned temp files
9. **Glossary size check**: Pass a >500KB glossary, confirm `ConfigurationError` is raised

---

## Decisions

- **Async-first, sync wrappers** — primary API is async; sync `transcribe_file()` uses `asyncio.run()` internally
- **No pydantic dependency** — stdlib `dataclasses` for settings to keep library lightweight. CLI also uses plain dataclasses (no pydantic-settings)
- **Backends are required params** — library never reads env vars; callers construct backends explicitly. `Settings.from_env()` classmethods are convenience helpers, not implicit behaviour
- **`requests` → `httpx`** — sole HTTP dependency, supports both sync and async
- **Retry stays in pipeline layer** — backends do a single HTTP call; `retry_with_backoff` is applied by the pipeline, not inside backends. This prevents retry logic duplication across providers
- **`max_retries` removed from `LLMBackend` protocol** — retry is a pipeline concern, not a backend interface concern
- **Audio processing stays sync** — PyAV is CPU-bound, wrapped in `asyncio.to_thread()` at the pipeline level
- **Prompts stay bundled** — no user customisation path (per user preference)
- **`_config.py` shim deleted** — it only re-exported from cli.py; tests/consumers update imports
- **Anthropic backend NOT implemented in this plan** — architecture makes it a single-module addition later
- **Scope boundary**: This plan covers architecture + the 14 adversarial findings. It does NOT add new features (no new providers, no streaming, no webhook callbacks)
- **Backend-agnostic chunking** — the pipeline always chunks long files when `duration > max_duration_before_split`, regardless of backend type. The current `isinstance(AzureTranscriptionBackend)` gate is removed. Whisper also benefits from chunking (reduces memory usage). Backends don't need a `supports_chunking` flag — chunking is a pipeline concern
- **`@runtime_checkable` async protocols** — `isinstance()` checks on protocols with `async def` methods will match sync implementations too (Python doesn't enforce async at the protocol level). This is acceptable; the type checker catches mismatches at call sites. Document this in `_protocols.py`
- **`format_whisper_output`** relocates to `backends/_whisper.py` alongside `WhisperTranscriptionBackend`

---

## Adversarial Findings Cross-Reference

| # | Finding | Resolution |
|---|---------|-----------|
| 1 | API key over HTTP | `_security.py::validate_https_url()` called in `Settings.from_env()` (earliest boundary) and backend constructors (defense in depth) |
| 2 | Prompt injection via template markers | `_prompts.py` uses single-pass `re.sub` replacement — user content is never scanned for markers |
| 3 | Temp file leak in `_convert_to_m4a` | Restructure with `try/finally` from `NamedTemporaryFile` creation |
| 4 | Temp dir leak in `_split_audio_file` | Add `try/finally` with `shutil.rmtree` |
| 5 | Thundering herd in parallel chunks | `asyncio.Semaphore` replaces unbounded `ThreadPoolExecutor` |
| 6 | `synthesise_text` ignores glossary | Documented as intentional — synthesis operates on final transcript |
| 7 | No input size validation | `_security.py::validate_input_size()` in pipeline |
| 8 | Hard-coded model names | Model names in `Settings` dataclasses with defaults |
| 9 | `sys.exit()` in library code | `_run()` raises `ConfigurationError`; only `main()` calls `sys.exit()` |
| 10 | Silent glossary correction failures | Return `CorrectionResult(text, was_corrected)` dataclass; pipeline honours `PipelineSettings.fail_on_correction_error` |
| 11 | No timeout on audio operations | Noted — PyAV frame processing has no clean timeout mechanism; documented as known limitation |
| 12 | `os.rmdir` won't remove non-empty dir | Replace with `shutil.rmtree` |
| 13 | No tests for audio cleanup/error paths | New `test_audio_cleanup.py` |
| 14 | `duration=None` skips chunking silently | Pipeline logs warning, attempts single-file transcription |

---

## Module Dependency Graph (Post-Architecture)

```
__init__.py  (re-exports + sync wrappers)
    └── _pipeline.py  (async orchestration)
            ├── _types.py  (TranscriptionResult, ChunkResult)
            ├── _protocols.py  (TranscriptionBackend, LLMBackend ABCs)
            ├── _prompts.py  (build_correction_prompt, build_synthesis_prompt)
            ├── _security.py  (validate_https_url, validate_input_size)
            ├── _retry.py  (async retry_with_backoff)
            ├── _audio.py  (sync audio processing, wrapped in to_thread)
            └── _exceptions.py  (exception hierarchy)

backends/
    ├── __init__.py  (factory functions)
    ├── _azure.py  (AzureTranscriptionBackend, AzureLLMBackend)
    │       ├── _settings.py::AzureTranscriptionSettings
    │       ├── _settings.py::AzureLLMSettings
    │       └── _security.py::validate_https_url
    └── _whisper.py  (WhisperTranscriptionBackend)
            └── _settings.py::WhisperSettings

cli.py  (entry point — reads env, builds backends, calls pipeline)
    ├── _settings.py  (Settings.from_env())
    ├── backends/  (factory functions)
    └── _pipeline.py  (transcribe, synthesise)
```

---

## Further Considerations

1. ~~**Sync wrapper strategy**~~ — **Resolved:** `_run_sync()` helper detects an existing event loop and raises a clear `RuntimeError` directing users to the async API. No `anyio` dependency needed.

2. ~~**Backend lifecycle management**~~ — **Resolved:** Backends implement `__aenter__`/`__aexit__` and offer `aclose()` as escape hatch. CLI uses `async with`. The sync wrapper in `__init__.py` doesn't need lifecycle management because `asyncio.run()` runs the full coroutine (including backend cleanup) to completion. Note: `__aenter__`/`__aexit__` are NOT added to the protocol types — they are concrete methods on backend implementations. The pipeline doesn't manage backend lifecycle; the caller does.

3. **Retry ownership for Anthropic** — Anthropic's SDK has built-in retry logic. When adding Anthropic backend, we need to decide: use their SDK's retries, or our pipeline-level retry? Recommendation: let the pipeline retry handle it uniformly; disable SDK-level retries in the Anthropic backend.

4. **Whisper model caching** — `asyncio.to_thread()` creates a new thread per call. With 15 parallel workers doing local Whisper, each thread would load the model independently. The `WhisperTranscriptionBackend` should load the model once in `__init__` and share it across `transcribe()` calls. This is safe because Whisper's `transcribe()` is stateless given a loaded model.
