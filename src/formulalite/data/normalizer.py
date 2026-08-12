"""Versioned lexical normalization for the FormulaLite LaTeX vocabulary."""

from __future__ import annotations

import unicodedata
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Final

NORMALIZER_VERSION: Final = "1.0.0"


@lru_cache(maxsize=1)
def baseline_tokens() -> tuple[str, ...]:
    """Return the fixed compact token table in its canonical order."""

    path = files("formulalite.data").joinpath("common_tokens.txt")
    tokens = tuple(line for line in path.read_text(encoding="utf-8").splitlines() if line)
    if len(tokens) != 683 or len(set(tokens)) != len(tokens):
        msg = "baseline token table must contain 683 unique tokens"
        raise RuntimeError(msg)
    return tokens


@lru_cache(maxsize=1)
def _compound_control_sequences() -> tuple[str, ...]:
    # Most commands are parsed by TeX's control-word/control-symbol rules. A small
    # number of baseline tokens intentionally include their environment argument.
    return tuple(
        sorted(
            (token for token in baseline_tokens() if token.startswith("\\") and "{" in token),
            key=len,
            reverse=True,
        )
    )


def _scan(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    compounds = _compound_control_sequences()

    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue

        if char != "\\":
            tokens.append(char)
            index += 1
            continue

        compound = next((item for item in compounds if text.startswith(item, index)), None)
        if compound is not None:
            tokens.append(compound)
            index += len(compound)
            continue

        if index + 1 >= len(text):
            tokens.append("\\")
            index += 1
            continue

        following = text[index + 1]
        if following.isalpha():
            end = index + 2
            while end < len(text) and text[end].isalpha():
                end += 1
            tokens.append(text[index:end])
            index = end
            continue

        # TeX control symbols consist of a backslash and one following character;
        # this also makes a row separator (\\\\) one stable token.
        tokens.append(text[index : index + 2])
        index += 2

    return tokens


def normalize(text: str) -> str:
    """Normalize LaTeX into the baseline's single-space lexical representation.

    The normalizer deliberately performs no semantic macro expansion. It separates
    commands, braces, scripts, operators, environment tokens, and ordinary symbols,
    making the operation deterministic and idempotent.
    """

    if not isinstance(text, str):
        msg = "LaTeX input must be a string"
        raise TypeError(msg)
    canonical = unicodedata.normalize("NFC", text).replace("\ufeff", "")
    return " ".join(_scan(canonical))


def normalizer_source_path() -> Path:
    """Expose the packaged vocabulary location for artifact tooling."""

    resource = files("formulalite.data").joinpath("common_tokens.txt")
    return Path(str(resource))
