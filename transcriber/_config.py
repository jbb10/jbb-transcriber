"""Backward-compatibility shim — config validation moved to cli.py.

Tests and legacy callers can still import from this module;
the canonical definitions live in ``transcriber.cli``.
"""

from __future__ import annotations

# Re-export so that ``from transcriber._config import ...`` keeps working.
from transcriber.cli import ValidatedConfig, validate_cli_config

__all__ = ["ValidatedConfig", "validate_cli_config"]
