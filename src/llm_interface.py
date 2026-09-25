"""Structural type for the LLM object this package needs.

Using a :class:`typing.Protocol` instead of importing
``llm_sdk.Small_LLM_Model`` directly keeps this module import-light (no
torch/transformers needed just to type-check or to run the unit tests) and
lets the test suite plug in a lightweight fake model with the exact same
public surface.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LlmLike(Protocol):
    """The subset of ``Small_LLM_Model``'s public API this package uses."""

    def encode(self, text: str) -> object:
        """Tokenize ``text`` (returns a 2-D tensor-like of input ids)."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits for the given id sequence."""

    def get_path_to_vocab_file(self) -> str:
        """Return the local path to the tokenizer's ``vocab.json``."""
