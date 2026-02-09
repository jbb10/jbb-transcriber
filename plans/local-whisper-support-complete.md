## Plan Complete: Local Whisper Transcription Support

Added local transcription using OpenAI Whisper as an alternative to Azure API. Users enable with `--local` flag, with optional `--model` and `--language` parameters. Azure credentials are no longer required for transcription when using local mode. Whisper is an optional dependency managed transparently by uv.

**Phases Completed:** 6 of 6
1. ✅ Phase 1: CLI Arguments & Config (pre-existing)
2. ✅ Phase 2: Whisper Import & Validation
3. ✅ Phase 3: Local Transcription Function
4. ✅ Phase 4: Integration & Routing
5. ✅ Phase 5: Config Validation Updates
6. ✅ Phase 6: Documentation & README

**All Files Created/Modified:**
- transcriber.py
- pyproject.toml
- README.md
- tests/test_negative.py

**Key Functions/Classes Added:**
- `import_whisper()` - Lazy import with helpful error message when not installed
- `format_whisper_output()` - Format Whisper segments into timestamped text
- `transcribe_audio_local()` - Transcribe audio using local Whisper model with GPU detection
- `validate_config()` - Updated to make Azure credentials optional in local mode
- `main()` - Updated to route to local or Azure transcription

**Test Coverage:**
- Total tests written: 11 (1 import test + 5 format tests + 5 validation tests)
- All tests passing: ✅ (36 total in test_negative.py)

**Recommendations for Next Steps:**
- Add integration test with actual Whisper model (e.g., tiny) on a short audio fixture
- Consider adding `--local` chunking support for very long files on low-memory devices
- Consider adding progress callback for Whisper transcription (e.g., percentage complete)
