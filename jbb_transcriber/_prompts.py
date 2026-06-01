"""Prompt loading and template rendering.

Prompt templates are bundled as package data files (``correction_prompt.md``,
``synthesis_prompt.md``).  Template markers like ``{{glossary}}`` are replaced
in a **single pass** to prevent double-substitution attacks (e.g. user content
containing ``{{transcript}}`` being treated as a marker).
"""

from __future__ import annotations

import re
from importlib.resources import files as pkg_files
from pathlib import Path

from jbb_transcriber._exceptions import PromptError


def _load_prompt(filename: str) -> str:
    """Load a prompt template bundled with the package.

    Args:
        filename: Name of the prompt file (e.g. ``"correction_prompt.md"``).

    Returns:
        The prompt template text.

    Raises:
        PromptError: If the template cannot be loaded.
    """
    try:
        return pkg_files("jbb_transcriber").joinpath(filename).read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError):
        # Fallback for editable installs / development
        prompt_path = Path(__file__).parent / filename
        try:
            return prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise PromptError(f"Prompt template not found: {filename}") from None


def _render_template(template: str, substitutions: dict[str, str]) -> str:
    """Replace template markers in a single pass (prevents double-substitution).

    Args:
        template: The template text containing markers.
        substitutions: Mapping of marker → replacement value.

    Returns:
        The rendered template.
    """
    if not substitutions:
        return template
    pattern = re.compile("|".join(re.escape(k) for k in substitutions))
    return pattern.sub(lambda m: substitutions[m.group(0)], template)


def build_correction_prompt(transcript: str, glossary_text: str) -> str:
    """Build a glossary-correction prompt from a transcript and glossary.

    Args:
        transcript: The raw transcription text.
        glossary_text: The glossary content.

    Returns:
        Complete prompt string ready to send to an LLM.
    """
    template = _load_prompt("correction_prompt.md")
    return _render_template(template, {"{{glossary}}": glossary_text, "{{transcript}}": transcript})


def build_synthesis_prompt(transcript: str) -> str:
    """Build a synthesis prompt from a transcript.

    Args:
        transcript: The transcription text.

    Returns:
        Complete prompt string ready to send to an LLM.
    """
    template = _load_prompt("synthesis_prompt.md")
    return _render_template(template, {"{{transcript}}": transcript})
