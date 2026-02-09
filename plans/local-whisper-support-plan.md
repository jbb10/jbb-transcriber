## Plan: Local Whisper Transcription Support

Add local transcription using OpenAI Whisper as an alternative to Azure API. Users enable with `--local` flag, with optional `--model` and `--language` parameters. Whisper becomes an optional uv dependency installed via `--with whisper`.

**Phases (6 phases)**

1. **Phase 1: CLI Arguments & Config**
    - **Objective:** Add `--local`, `--model`, and `--language` arguments to CLI and integrate with ValidatedConfig
    - **Files/Functions to Modify/Create:**
      - [transcriber.py](../transcriber.py) - `ValidatedConfig` dataclass (add `local_mode`, `whisper_model`, `language` fields)
      - [transcriber.py](../transcriber.py) - argparse section (add new argument definitions)
    - **Tests to Write:**
      - `test_cli_local_args_parsed` - verify --local, --model, --language parsed correctly
      - `test_cli_local_requires_model_only_with_flag` - --model without --local shows warning/ignored
      - `test_config_includes_local_fields` - ValidatedConfig has new fields
    - **Steps:**
        1. Write tests for CLI argument parsing with new local flags
        2. Run tests to verify they fail
        3. Add new fields to `ValidatedConfig` dataclass
        4. Add argparse arguments: `--local`/`-l` (flag), `--model` (default "base"), `--language` (optional)
        5. Run tests to verify they pass
        6. Run `make lint` to ensure code quality

2. **Phase 2: Whisper Import & Validation**
    - **Objective:** Add optional Whisper import with helpful error message if missing, validate model names
    - **Files/Functions to Modify/Create:**
      - [transcriber.py](../transcriber.py) - new function `import_whisper()` for lazy import
      - [transcriber.py](../transcriber.py) - `validate_config()` to check Whisper availability when `--local`
      - [pyproject.toml](../pyproject.toml) - add `[project.optional-dependencies]` section
    - **Tests to Write:**
      - `test_validate_config_local_without_whisper_fails` - helpful error when Whisper not installed
      - `test_validate_config_local_invalid_model_fails` - reject invalid model names
      - `test_validate_config_local_valid_passes` - accepts valid local configuration
    - **Steps:**
        1. Write tests for Whisper import validation and model name validation
        2. Run tests to verify they fail
        3. Create `import_whisper()` function that returns module or raises with install instructions
        4. Add model validation against known Whisper model names
        5. Update `validate_config()` to call these validations when `args.local` is True
        6. Add optional-dependencies to pyproject.toml: `whisper = ["openai-whisper>=20231117"]`
        7. Run tests to verify they pass
        8. Run `make lint`

3. **Phase 3: Local Transcription Function**
    - **Objective:** Implement `transcribe_audio_local()` function using Whisper API
    - **Files/Functions to Modify/Create:**
      - [transcriber.py](../transcriber.py) - new function `transcribe_audio_local(audio_path, config)` 
      - [transcriber.py](../transcriber.py) - new function `format_whisper_output(result)` for output conversion
    - **Tests to Write:**
      - `test_transcribe_audio_local_basic` - basic transcription returns text with timestamps
      - `test_format_whisper_output_segments` - segments formatted as `[0.00s - 5.23s] text`
      - `test_transcribe_audio_local_with_language` - explicit language passed to Whisper
      - `test_transcribe_audio_local_model_loading` - model loads correctly
    - **Steps:**
        1. Write tests for local transcription (skip if Whisper not installed)
        2. Run tests to verify they fail
        3. Implement `transcribe_audio_local()`: load model, call transcribe, return result
        4. Implement `format_whisper_output()`: convert Whisper segments to `[start - end] text` format
        5. Add logging for model loading, language detection, device used
        6. Run tests to verify they pass
        7. Run `make lint`

4. **Phase 4: Integration & Routing**
    - **Objective:** Route transcription to local or Azure based on config, handle glossary as initial_prompt
    - **Files/Functions to Modify/Create:**
      - [transcriber.py](../transcriber.py) - refactor `transcribe_audio()` to delegate to provider-specific functions
      - [transcriber.py](../transcriber.py) - update `process_audio()` to pass local config through
    - **Tests to Write:**
      - `test_transcribe_routes_to_local_when_flag_set` - --local routes to local function
      - `test_transcribe_routes_to_azure_by_default` - default remains Azure API
      - `test_local_with_glossary_uses_initial_prompt` - glossary terms passed as Whisper initial_prompt
    - **Steps:**
        1. Write tests for routing logic between local and Azure
        2. Run tests to verify they fail
        3. Refactor `transcribe_audio()` to check `config.local_mode` and delegate
        4. When local + glossary: read glossary file and pass as `initial_prompt`
        5. Update parallel processing to work with local mode
        6. Run tests to verify they pass
        7. Run `make lint`

5. **Phase 5: Config Validation Updates**
    - **Objective:** Make Azure API credentials optional when using local mode
    - **Files/Functions to Modify/Create:**
      - [transcriber.py](../transcriber.py) - update `validate_config()` to not require Azure env vars when `--local`
    - **Tests to Write:**
      - `test_local_mode_no_azure_credentials_required` - --local works without AZURE_TRANSCRIBE_* env vars
      - `test_azure_mode_still_requires_credentials` - default mode still validates Azure credentials
      - `test_local_with_correction_requires_azure_text_api` - --local --glossary with LLM correction needs Azure
    - **Steps:**
        1. Write tests for credential requirement differences
        2. Run tests to verify they fail
        3. Update `validate_config()` to skip Azure credential check when `config.local_mode` is True
        4. Add check: if `--local` with `--glossary` and synthesis enabled, still need Azure LLM credentials
        5. Run tests to verify they pass
        6. Run `make lint`

6. **Phase 6: Documentation & README**
    - **Objective:** Update README with local mode documentation, installation instructions, model comparison
    - **Files/Functions to Modify/Create:**
      - [README.md](../README.md) - add "Local Whisper Mode" section
      - [README.md](../README.md) - update installation instructions for `--with whisper`
    - **Tests to Write:**
      - (No automated tests - documentation review only)
    - **Steps:**
        1. Add "Local Transcription Mode" section to README explaining:
           - Installation: `uv tool install --with whisper transcriber`
           - Usage: `transcribe --local audio.mp3 output.txt`
           - Model selection table (tiny→large with size/speed tradeoffs)
           - Language options (auto-detect vs explicit)
           - Limitations (no speaker diarization)
        2. Update existing usage examples to show local mode option
        3. Add note about FFmpeg requirement (already needed for current functionality)
        4. Run `make lint` for any markdown linting

**Open Questions (2 questions, ~10-20 words each)**
1. **GPU logging?** Should we log "Using CUDA" vs "Using CPU" when loading model? → Yes, helpful for performance expectations
2. **Large model warning?** Warn users if selecting large/turbo models about download size (~6-10GB)? → Yes, add warning for models >500MB
