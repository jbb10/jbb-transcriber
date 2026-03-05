# Changelog

All notable changes to this project will be documented in this file.
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



