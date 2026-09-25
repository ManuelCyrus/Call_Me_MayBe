"""Unit tests for the constrained-decoding primitives, using
``FakeSmallLLMModel`` so no network access or real weights are needed.
"""
from __future__ import annotations

from src.constrained import Generator, GenerationError, VocabIndex
from tests.fake_model import FakeSmallLLMModel


def make_generator(script: list[str]) -> Generator:
    model = FakeSmallLLMModel(script)
    vocab = VocabIndex.from_model(model)
    return Generator(model, vocab, input_ids=[])


def test_force_appends_deterministic_tokens() -> None:
    gen = make_generator(script=[])
    gen.force('{"a": ')
    assert gen.model.decode(gen.ids) == '{"a": '  # type: ignore[attr-defined]


def test_choose_enum_multi_token_names() -> None:
    # "fn_" is shared by both candidates and costs no model call; the only
    # real decision is "a" vs "b".
    gen = make_generator(script=["b"])
    chosen = gen.choose_enum(['fn_a"', 'fn_b"'])
    assert chosen == 'fn_b"'


def test_choose_enum_single_candidate_needs_no_model_call() -> None:
    gen = make_generator(script=[])  # nothing scripted: would fail if used
    chosen = gen.choose_enum(['only_option"'])
    assert chosen == 'only_option"'


def test_generate_number_positive_integer_like() -> None:
    gen = make_generator(script=["2", "3", "}"])
    value = gen.generate_number(exit_text="}", integer_only=False)
    assert value == 23
    assert isinstance(value, int)


def test_generate_number_decimal_and_negative() -> None:
    gen = make_generator(script=["-", "4", ".", "5", "}"])
    value = gen.generate_number(exit_text="}", integer_only=False)
    assert value == -4.5


def test_generate_number_integer_only_rejects_dot() -> None:
    # "." is not in the allowed char set for an integer DFA, so once "7"
    # is accepting the model's wish for "." is masked out; only the exit
    # path remains and the value stops at 7.
    gen = make_generator(script=["7", "."])
    value = gen.generate_number(exit_text="}", integer_only=True)
    assert value == 7


def test_generate_string_basic_word() -> None:
    gen = make_generator(script=["s", "h", "r", "e", "k", '"'])
    text = gen.generate_string(exit_text='"' + ", ")
    assert text == "shrek"


def test_generate_string_can_be_empty() -> None:
    gen = make_generator(script=['"'])
    text = gen.generate_string(exit_text='"}')
    assert text == ""


def test_generate_number_choice_two_items_then_stop() -> None:
    gen = make_generator(script=["1", ",", "2", "]"])
    items = []
    for _ in range(5):
        value, done = gen.generate_number_choice(", ", "]", integer_only=True)
        items.append(value)
        if done:
            break
    assert items == [1, 2]


def test_generate_string_choice_stops_immediately_when_offered() -> None:
    # A string may be empty, so "end" is already on offer at step 0. Both
    # exit texts start with the same closing quote; the second scripted
    # token ("]") resolves that tie in favour of ending the array.
    gen = make_generator(script=['"', "]"])
    value, done = gen.generate_string_choice('", ', '"]')
    assert value == ""
    assert done is True


def test_argmax_raises_on_empty_pool() -> None:
    gen = make_generator(script=[])
    try:
        gen._generate_value(  # noqa: SLF001 - internal, testing directly
            allowed_ids=frozenset(),
            is_accepting=lambda _t: False,
            step=lambda _t, _a: None,
            exit_options=[[0]],
        )
    except GenerationError:
        pass
    else:
        raise AssertionError("expected GenerationError")
