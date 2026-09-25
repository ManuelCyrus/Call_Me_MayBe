"""Tests for input/output file handling and its error messages."""
from __future__ import annotations

import json

import pytest

from src.io_utils import ConfigError, load_functions, load_prompts, write_output
from src.schemas import OutputEntry


def test_load_functions_happy_path(tmp_path: "object") -> None:
    path = tmp_path / "functions.json"  # type: ignore[operator]
    path.write_text(json.dumps([
        {"name": "fn_x", "description": "d", "parameters": {"a": {"type": "number"}}}
    ]))
    functions = load_functions(str(path))
    assert len(functions) == 1
    assert functions[0].name == "fn_x"


def test_load_functions_missing_file(tmp_path: "object") -> None:
    missing = tmp_path / "nope.json"  # type: ignore[operator]
    with pytest.raises(ConfigError, match="not found"):
        load_functions(str(missing))


def test_load_functions_invalid_json(tmp_path: "object") -> None:
    path = tmp_path / "bad.json"  # type: ignore[operator]
    path.write_text("{not valid json")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_functions(str(path))


def test_load_functions_wrong_shape(tmp_path: "object") -> None:
    path = tmp_path / "wrong.json"  # type: ignore[operator]
    path.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ConfigError, match="JSON array"):
        load_functions(str(path))


def test_load_prompts_happy_path(tmp_path: "object") -> None:
    path = tmp_path / "prompts.json"  # type: ignore[operator]
    path.write_text(json.dumps([{"prompt": "hi"}]))
    prompts = load_prompts(str(path))
    assert prompts[0].prompt == "hi"


def test_write_output_creates_directory(tmp_path: "object") -> None:
    out_path = tmp_path / "sub" / "dir" / "out.json"  # type: ignore[operator]
    entries = [OutputEntry(prompt="p", fn_name="fn_x", args={"a": 1})]
    write_output(str(out_path), entries)
    saved = json.loads(out_path.read_text())
    assert saved == [{"prompt": "p", "fn_name": "fn_x", "args": {"a": 1}}]
