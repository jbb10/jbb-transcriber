# Changelog

All notable changes to this project will be documented in this file.

## [5.0.0] - 2026-06-01

### Refactoring
- Rename project to `jbb-transcriber`; import package is now `jbb_transcriber` (**BREAKING**)
- Rename environment variables `TRANSCRIBER_*` to `JBB_TRANSCRIBER_*` (**BREAKING**)

## [4.0.1] - 2026-03-22

### Bug Fixes
- Reduce chunk duration and add env-var tuning for Azure App Service timeout


## [4.0.0] - 2026-03-22

### Features
- Unified env vars and pipeline reliability improvements (**BREAKING**)


## [3.1.0] - 2026-03-11

### Bug Fixes
- Use function-scoped fixtures for async backends


### Features
- Add --yes flag to release script for CI/AI automation
- Async-first provider-agnostic architecture with DI and httpx
- Auto-detect text files for synthesis in CLI


## [3.0.0] - 2026-03-05

### Features
- Add LLMError exception and smart retry with transient error classification (**BREAKING**)


## [2.1.0] - 2026-03-04

### Features
- Add configurable timeout, retry, progress callback, chunk duration and structured ChunkResult


## [2.0.0] - 2026-03-01

### Features
- Refactor monolith into modular package with protocol-based DI (**BREAKING**)


## [1.1.0] - 2026-02-28

### Features
- Add --synthesise-only flag to generate synthesis from existing transcript


## [1.0.0] - 2026-02-09

### Documentation
- Add local Whisper transcription documentation


### Features
- Make whisper a core dependency and remove --language flag (**BREAKING**)
- Route transcription to local Whisper or Azure API
- Add local Whisper transcription function
- Add lazy Whisper import and optional dependency


## [0.1.1] - 2026-02-09

### Bug Fixes
- Add type ignore comments for PyAV library type checking


### Style
- Fix linting and formatting issues



