.PHONY: install install-dev install-local test test-fast test-unit test-negative build clean lint fix release help

help:
	@echo "Available targets:"
	@echo "  install       - Install package dependencies"
	@echo "  install-dev   - Install package with dev dependencies (ruff, pyright, pytest)"
	@echo "  install-local - Install package with local Whisper support"
	@echo "  test          - Run all tests"
	@echo "  test-fast     - Run tests that don't require Azure/audio files"
	@echo "  test-unit     - Run unit tests only (no Azure API calls)"
	@echo "  test-negative - Run only negative/error handling tests"
	@echo "  build         - Build package (wheel and sdist)"
	@echo "  clean         - Remove build artifacts and caches"
	@echo "  lint          - Check code (ruff + pyright)"
	@echo "  fix           - Auto-fix code style issues"
	@echo "  release       - Release a new version (auto-detects bump from commits)"

install:
	uv pip install -e .

install-dev:
	uv sync --group dev

install-local:
	uv pip install -e ".[local]"

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/test_negative.py tests/test_cli.py::TestCLIErrorHandling::test_cli_help tests/test_cli.py::TestCLIErrorHandling::test_cli_missing_input_file -v

test-unit:
	uv run pytest tests/test_negative.py tests/test_api.py -v

test-negative:
	uv run pytest tests/test_negative.py -v

build:
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf __pycache__ tests/__pycache__ jbb_transcriber/__pycache__
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

fix:
	uv run ruff check --fix .
	uv run ruff format .

release:
	./scripts/release.sh
