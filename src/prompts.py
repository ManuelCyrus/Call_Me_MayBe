"""Build the natural-language prompt shown to the model before generation.

The model never has to produce JSON syntax on its own -- that part is
forced (see :mod:`src.constrained`). This prompt only has to give it
enough context to make good *content* decisions: which function fits the
request, and what the argument values should be.
"""
from __future__ import annotations

from src.schemas import FunctionDefinition

_PREAMBLE = (
    "You are a function-calling assistant. Read the user's request and the "
    "list of available functions below, then decide which single function "
    "answers the request and what its argument values should be.\n\n"
    "Available functions:\n"
)


def _describe_function(func: FunctionDefinition) -> str:
    params = ", ".join(
        f"{name}: {spec.type}" for name, spec in func.parameters.items()
    )
    return f"- {func.name}({params}): {func.description}"


def build_prompt(user_prompt: str, functions: list[FunctionDefinition]) -> str:
    """Return the full prompt text to feed the model for one request."""
    lines = [_PREAMBLE]
    lines.extend(_describe_function(f) for f in functions)
    lines.append("")
    lines.append(f"User request: {user_prompt}")
    lines.append("")
    return "\n".join(lines)
