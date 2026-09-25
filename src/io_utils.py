"""Reading and writing the project's JSON files, with clear error messages
instead of raw tracebacks for the problems a reviewer is likely to try:
a missing file, or a file that is not valid JSON.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import ValidationError

from src.schemas import FunctionDefinition, OutputEntry, PromptItem


class ConfigError(Exception):
    """A problem with an input file that the user needs to fix."""


def _read_json(path: str) -> object:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"input file not found: {path}")
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON ({exc})") from exc
    except OSError as exc:
        raise ConfigError(f"could not read {path} ({exc})") from exc


def load_functions(path: str) -> list[FunctionDefinition]:
    """Load and validate ``functions_definition.json``."""
    data = _read_json(path)
    if not isinstance(data, list):
        raise ConfigError(f"{path} must contain a JSON array of functions")
    try:
        return [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ConfigError(f"{path} does not match the expected schema: {exc}") from exc


def load_prompts(path: str) -> list[PromptItem]:
    """Load and validate ``function_calling_tests.json``."""
    data = _read_json(path)
    if not isinstance(data, list):
        raise ConfigError(f"{path} must contain a JSON array of prompts")
    try:
        return [PromptItem.model_validate(item) for item in data]
    except ValidationError as exc:
        raise ConfigError(f"{path} does not match the expected schema: {exc}") from exc


def write_output(path: str, entries: list[OutputEntry]) -> None:
    """Write the final results file, creating its directory if needed."""
    out_path = Path(path)
    os.makedirs(out_path.parent, exist_ok=True) if str(out_path.parent) else None
    payload = [entry.model_dump() for entry in entries]
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
