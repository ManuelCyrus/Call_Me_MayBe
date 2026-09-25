"""End-to-end tests of the full JSON-generation grammar (schema_gen.py)."""
from __future__ import annotations

import pytest

from src.constrained import Generator, VocabIndex
from src.schema_gen import generate_function_call
from src.schemas import FunctionDefinition, ParameterSpec
from tests.fake_model import FakeSmallLLMModel

ADD_NUMBERS = FunctionDefinition(
    name="fn_add_numbers",
    description="Add two numbers together and return their sum.",
    parameters={
        "a": ParameterSpec(type="number"),
        "b": ParameterSpec(type="number"),
    },
)
GREET = FunctionDefinition(
    name="fn_greet",
    description="Generate a greeting message for a person by name.",
    parameters={"name": ParameterSpec(type="string")},
)
REVERSE = FunctionDefinition(
    name="fn_reverse_string",
    description="Reverse a string and return the reversed result.",
    parameters={"s": ParameterSpec(type="string")},
)
SET_FIRMWARE = FunctionDefinition(
    name="fn_set_firmware",
    description="Flash a device with one of the supported firmware builds.",
    parameters={
        "firmware": ParameterSpec(type="string", enum=["stable", "beta", "nightly"])
    },
)
LIST_NUMBERS = FunctionDefinition(
    name="fn_sum_list",
    description="Sum a list of numbers.",
    parameters={
        "values": ParameterSpec(type="array", items=ParameterSpec(type="integer"))
    },
)


def make_generator(script: list[str]) -> Generator:
    model = FakeSmallLLMModel(script)
    vocab = VocabIndex.from_model(model)
    return Generator(model, vocab, input_ids=[])


def test_picks_second_function_and_fills_numbers() -> None:
    functions = [ADD_NUMBERS, GREET]
    # Shared prefix "fn_" costs no model call (only one candidate is
    # consistent with it); the first real decision is "a" vs "g".
    script = ["g"] + ["s", "h", "r", "e", "k", '"']
    gen = make_generator(script)
    name, args = generate_function_call(gen, functions)
    assert name == "fn_greet"
    assert args == {"name": "shrek"}


def test_add_numbers_end_to_end() -> None:
    functions = [ADD_NUMBERS, GREET]
    script = ["a"] + ["2", "6", "5", ","] + ["3", "4", "5", "}"]
    gen = make_generator(script)
    name, args = generate_function_call(gen, functions)
    assert name == "fn_add_numbers"
    assert args == {"a": 265, "b": 345}
    # The token ids we accumulated, decoded, must be exactly valid JSON.
    import json

    full_text = gen.model.decode(gen.ids)  # type: ignore[attr-defined]
    parsed = json.loads(full_text)
    assert parsed == {"fn_name": "fn_add_numbers", "args": {"a": 265, "b": 345}}


def test_reverse_string_single_param() -> None:
    functions = [ADD_NUMBERS, REVERSE]
    script = ["r"] + ["h", "e", "l", "l", "o", '"']
    gen = make_generator(script)
    name, args = generate_function_call(gen, functions)
    assert name == "fn_reverse_string"
    assert args == {"s": "hello"}


def test_enum_parameter_restricts_to_allowed_values() -> None:
    functions = [SET_FIRMWARE]
    # Single function: no branching needed to pick it. The three enum
    # options share the opening quote, then diverge at "s"/"b"/"n".
    script = ["b"]
    gen = make_generator(script)
    name, args = generate_function_call(gen, functions)
    assert name == "fn_set_firmware"
    assert args["firmware"] in {"stable", "beta", "nightly"}
    assert args["firmware"] == "beta"


def test_array_of_integers_bonus_nested_support() -> None:
    functions = [LIST_NUMBERS]
    script = ["1", "0", ",", "2", "0", "]"]
    gen = make_generator(script)
    name, args = generate_function_call(gen, functions)
    assert name == "fn_sum_list"
    assert args == {"values": [10, 20]}


LIST_GREETINGS = FunctionDefinition(
    name="fn_greet_many",
    description="Greet a list of people by name.",
    parameters={
        "names": ParameterSpec(type="array", items=ParameterSpec(type="string"))
    },
)


def test_array_of_strings_end_to_end() -> None:
    # Both the "one more item" and "end the array" continuations start by
    # closing the current string's quote, so this also exercises the
    # shared-prefix tie-break inside array-of-string generation.
    functions = [LIST_GREETINGS]
    script = ["a", "n", "a", '"', ",", "b", "o", "b", '"', "]"]
    gen = make_generator(script)
    name, args = generate_function_call(gen, functions)
    assert name == "fn_greet_many"
    assert args == {"names": ["ana", "bob"]}

    import json

    full_text = gen.model.decode(gen.ids)  # type: ignore[attr-defined]
    assert json.loads(full_text) == {
        "fn_name": "fn_greet_many",
        "args": {"names": ["ana", "bob"]},
    }


def test_array_of_enum_items_is_rejected_cleanly() -> None:
    bad_fn = FunctionDefinition(
        name="fn_bad",
        description="Not supported: array of enum-restricted items.",
        parameters={
            "flags": ParameterSpec(
                type="array", items=ParameterSpec(type="string", enum=["a", "b"])
            )
        },
    )
    gen = make_generator([])
    with pytest.raises(ValueError):
        generate_function_call(gen, [bad_fn])


def test_empty_function_list_raises() -> None:
    gen = make_generator([])
    with pytest.raises(ValueError):
        generate_function_call(gen, [])
