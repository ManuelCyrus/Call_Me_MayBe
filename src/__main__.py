"""CLI entry point.

Usage
-----
uv run python -m src [--functions_definition <file>] [--input <file>]
                      [--output <file>] [--model <hf model id>]
"""
from __future__ import annotations

import argparse
import sys

from src.engine import FunctionCaller
from src.io_utils import ConfigError, load_functions, load_prompts, write_output
from src.llm_interface import LlmLike
from src.schemas import OutputEntry

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=(
            "Translate natural-language prompts into structured function "
            "calls using constrained decoding."
        ),
    )
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face model id passed to Small_LLM_Model (bonus: any "
            "model exposing the same SDK interface works)."
        ),
    )
    return parser.parse_args(argv)


def _load_model(model_name: str) -> LlmLike:
    """Import and construct the SDK model lazily.

    Importing ``llm_sdk`` (and, transitively, torch) only when actually
    needed keeps ``--help`` and the unit tests fast, and turns a missing
    dependency into a clear message instead of an import-time crash.
    """
    try:
        # mypy note: static analysis can resolve "llm_sdk" to the vendored
        # source folder that sits beside src/ (itself just a namespace
        # package with no __init__.py) instead of the distribution uv
        # actually installs from it, and reports a false "no attribute"
        # here. At runtime and under `uv run`, the installed package
        # resolves correctly (see README: "Known limitations").
        from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
    except ImportError as exc:
        raise ConfigError(
            "could not import llm_sdk. Make sure it is installed "
            f"(see README: 'uv sync'). Details: {exc}"
        ) from exc
    try:
        model: LlmLike = Small_LLM_Model(model_name=model_name)
        return model
    except Exception as exc:  # noqa: BLE001 - surfaced as a clear message
        raise ConfigError(
            f"could not load model {model_name!r}: {exc}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    """Run the full pipeline. Returns a process exit code."""
    args = parse_args(argv)

    try:
        functions = load_functions(args.functions_definition)
        prompts = load_prompts(args.input)
        model = _load_model(args.model)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not functions:
        print(
            "Warning: no functions defined in "
            f"{args.functions_definition}; every prompt will be skipped.",
            file=sys.stderr,
        )

    caller = FunctionCaller(model, functions)
    results: list[OutputEntry] = []
    skipped = 0
    for item in prompts:
        try:
            results.append(caller.call(item.prompt))
        except Exception as exc:  # noqa: BLE001 - one bad prompt must not
            # stop the whole run; it is reported and skipped instead.
            skipped += 1
            print(
                f"Warning: skipped prompt {item.prompt!r}: {exc}",
                file=sys.stderr,
            )

    try:
        write_output(args.output, results)
    except OSError as exc:
        print(f"Error: could not write {args.output}: {exc}", file=sys.stderr)
        return 1

    print(
        f"Processed {len(prompts)} prompt(s): {len(results)} succeeded, "
        f"{skipped} skipped. Output written to {args.output}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
