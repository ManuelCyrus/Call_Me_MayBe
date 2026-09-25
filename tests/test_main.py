"""End-to-end tests of the CLI (src/__main__.py), using the fake model so
no network access or real weights are required.
"""
from __future__ import annotations

import json

import pytest

import src.__main__ as main_module
from tests.fake_model import FakeSmallLLMModel


def _write(path: "object", data: object) -> None:
    path.write_text(json.dumps(data))  # type: ignore[attr-defined]


def test_main_happy_path(tmp_path: "object", monkeypatch: "pytest.MonkeyPatch") -> None:
    functions_path = tmp_path / "functions.json"  # type: ignore[operator]
    prompts_path = tmp_path / "prompts.json"  # type: ignore[operator]
    output_path = tmp_path / "out" / "results.json"  # type: ignore[operator]

    _write(functions_path, [
        {
            "name": "fn_greet",
            "description": "Greet someone.",
            "parameters": {"name": {"type": "string"}},
        }
    ])
    _write(prompts_path, [{"prompt": "Greet shrek"}])

    # Single function, single string param: "s","h","r","e","k" then close.
    fake = FakeSmallLLMModel(["s", "h", "r", "e", "k", '"'])
    monkeypatch.setattr(main_module, "_load_model", lambda _name: fake)

    exit_code = main_module.main([
        "--functions_definition", str(functions_path),
        "--input", str(prompts_path),
        "--output", str(output_path),
    ])

    assert exit_code == 0
    saved = json.loads(output_path.read_text())
    assert saved == [
        {"prompt": "Greet shrek", "fn_name": "fn_greet", "args": {"name": "shrek"}}
    ]


def test_main_missing_input_file_reports_error(
    tmp_path: "object", monkeypatch: "pytest.MonkeyPatch", capsys: "pytest.CaptureFixture[str]"
) -> None:
    functions_path = tmp_path / "functions.json"  # type: ignore[operator]
    _write(functions_path, [])
    monkeypatch.setattr(main_module, "_load_model", lambda _name: FakeSmallLLMModel([]))

    exit_code = main_module.main([
        "--functions_definition", str(functions_path),
        "--input", str(tmp_path / "missing.json"),  # type: ignore[operator]
        "--output", str(tmp_path / "out.json"),  # type: ignore[operator]
    ])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_main_skips_failing_prompt_without_crashing(
    tmp_path: "object", monkeypatch: "pytest.MonkeyPatch", capsys: "pytest.CaptureFixture[str]"
) -> None:
    functions_path = tmp_path / "functions.json"  # type: ignore[operator]
    prompts_path = tmp_path / "prompts.json"  # type: ignore[operator]
    output_path = tmp_path / "results.json"  # type: ignore[operator]

    # No functions available at all: every prompt must fail to produce a
    # call, but the program must still exit cleanly with an empty result.
    _write(functions_path, [])
    _write(prompts_path, [{"prompt": "anything"}])
    monkeypatch.setattr(main_module, "_load_model", lambda _name: FakeSmallLLMModel([]))

    exit_code = main_module.main([
        "--functions_definition", str(functions_path),
        "--input", str(prompts_path),
        "--output", str(output_path),
    ])

    assert exit_code == 0
    saved = json.loads(output_path.read_text())
    assert saved == []
    captured = capsys.readouterr()
    assert "skipped prompt" in captured.err
