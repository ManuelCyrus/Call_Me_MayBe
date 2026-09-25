"""Turn a function's parameter schema into forced JSON text plus
model-driven decisions, using the primitives in :mod:`src.constrained`.

``closing`` is threaded through every call: it is the exact literal text
that must follow a value once its own content is finished (for example
``', "'`` when another sibling key comes next, or ``'}'`` chained with
whatever the parent needed). Composing it this way means each function
only ever has to worry about its own brace or bracket, never about how
deep it currently is nested.
"""
from __future__ import annotations

import json
from typing import Any

from src.constrained import Generator
from src.schemas import FunctionDefinition, ParameterSpec

_NUMBER_TYPES = {"number", "float", "double"}
_INTEGER_TYPES = {"integer", "int"}
_BOOLEAN_TYPES = {"boolean", "bool"}


def _properties_of(spec: ParameterSpec) -> dict[str, ParameterSpec] | None:
    """Read a nested ``properties`` schema for an ``object`` parameter, if
    the function definition provided one as an extra field."""
    extra = spec.model_extra
    if not isinstance(extra, dict):
        return None
    raw = extra.get("properties")
    if not isinstance(raw, dict):
        return None
    result: dict[str, ParameterSpec] = {}
    for key, value in raw.items():
        result[key] = value if isinstance(value, ParameterSpec) else ParameterSpec(**value)
    return result


def generate_value(gen: Generator, spec: ParameterSpec, closing: str) -> Any:
    """Generate one value for ``spec`` and append ``closing`` after it."""
    if spec.enum:
        literal_texts = [json.dumps(v) + closing for v in spec.enum]
        chosen_full = gen.choose_enum(literal_texts)
        for value in spec.enum:
            if json.dumps(value) + closing == chosen_full:
                return value
        # Unreachable in practice: choose_enum always returns one of the
        # candidates it was given.
        return spec.enum[0]

    kind = spec.type

    if kind in _NUMBER_TYPES:
        return gen.generate_number(exit_text=closing, integer_only=False)

    if kind in _INTEGER_TYPES:
        return gen.generate_number(exit_text=closing, integer_only=True)

    if kind in _BOOLEAN_TYPES:
        chosen_full = gen.choose_enum([f"true{closing}", f"false{closing}"])
        return chosen_full.startswith("true")

    if kind == "array":
        return generate_array(gen, spec.items or ParameterSpec(type="string"), closing)

    if kind == "object":
        properties = _properties_of(spec)
        if properties:
            return generate_object(gen, properties, outer_closing=closing)
        # No sub-schema was provided: fall back to an empty object rather
        # than guessing at unconstrained content (see README limitations).
        gen.force("{}" + closing)
        return {}

    # "string", and any type name we do not otherwise recognize: treated
    # as free-form text so an unusual schema still produces an answer.
    gen.force('"')
    return gen.generate_string(exit_text='"' + closing)


def generate_array(gen: Generator, item_spec: ParameterSpec, closing: str) -> list[Any]:
    """Generate ``[item, item, ...]`` for a primitive-typed ``items`` schema.

    Each element and the "one more item, or close the array" decision are
    resolved by the *same* masked-logit comparison (see
    :meth:`Generator.generate_number_choice` /
    :meth:`Generator.generate_string_choice`), so no extra model call is
    spent purely on bookkeeping.

    Only ``number``, ``integer`` and ``string`` items are supported: this
    covers the realistic "list of values" case described as a bonus in the
    subject. Arrays of enums, booleans, objects or nested arrays raise a
    clear error instead of silently guessing at an ad hoc encoding, so a
    schema this tool cannot honour never turns into wrong output.
    """
    gen.force("[")
    kind = item_spec.type
    items: list[Any] = []
    end_text = "]" + closing
    if item_spec.enum:
        raise ValueError("arrays of enum-restricted items are not supported")
    for _ in range(20):
        value: Any
        if kind in _NUMBER_TYPES:
            value, done = gen.generate_number_choice(", ", end_text, integer_only=False)
        elif kind in _INTEGER_TYPES:
            value, done = gen.generate_number_choice(", ", end_text, integer_only=True)
        elif kind == "string":
            gen.force('"')
            value, done = gen.generate_string_choice('", ', '"' + end_text)
        else:
            raise ValueError(
                f"arrays of '{kind}' items are not supported; only number, "
                "integer and string item types are"
            )
        items.append(value)
        if done:
            return items
    raise ValueError("array exceeded the maximum supported number of items")


def generate_object(
    gen: Generator, properties: dict[str, ParameterSpec], outer_closing: str
) -> dict[str, Any]:
    """Generate ``{"k1": v1, "k2": v2, ...}`` for an ordered property map."""
    gen.force("{")
    items = list(properties.items())
    if not items:
        gen.force("}" + outer_closing)
        return {}
    result: dict[str, Any] = {}
    for index, (key, spec) in enumerate(items):
        is_last = index == len(items) - 1
        own_closing = ("}" + outer_closing) if is_last else ", "
        gen.force(f'"{key}": ')
        result[key] = generate_value(gen, spec, own_closing)
    return result


def generate_function_call(
    gen: Generator, functions: list[FunctionDefinition]
) -> tuple[str, dict[str, Any]]:
    """Generate ``{"fn_name": "...", "args": {...}}`` end to end.

    The function name is chosen by the model from the closed set of
    available names; the arguments are then generated against exactly
    that function's own parameter schema.
    """
    if not functions:
        raise ValueError("no functions available to choose from")
    gen.force('{"fn_name": "')
    names = [f.name for f in functions]
    chosen_full = gen.choose_enum([f'{name}"' for name in names])
    chosen_name = chosen_full[: -1]  # drop the trailing quote we appended
    gen.force(', "args": ')
    func = next(f for f in functions if f.name == chosen_name)
    args = generate_object(gen, func.parameters, outer_closing="}")
    return chosen_name, args
