"""A tiny fake model with the same public surface as ``Small_LLM_Model``.

It lets the test suite exercise the constrained-decoding logic in
``src.constrained`` and ``src.schema_gen`` deterministically, with no
network access and no GPU/torch dependency: exactly what this sandbox and
a CI runner without internet access both need.
"""
from __future__ import annotations

import json
import string
import tempfile
from pathlib import Path

from src.vocab_utils import encode_vocab_token

_ALPHABET = string.ascii_letters + string.digits + " _\"{}[]():,.-'?!\n/;%+=<>"


class FakeSmallLLMModel:
    """Deterministic stand-in for ``Small_LLM_Model``.

    Parameters
    ----------
    script : list[str]
        The token *texts* the fake model "wants" to produce, in order.
        Each call to :meth:`get_logits_from_input_ids` consumes the next
        entry and returns logits that strongly favour whichever known
        character equals it (letting the real masking logic in
        ``src.constrained`` decide whether that preference is actually
        allowed at this point in the grammar).
    """

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self._call_count = 0
        self.char_to_id = {ch: i for i, ch in enumerate(_ALPHABET)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self._vocab_dir = tempfile.mkdtemp(prefix="fake_vocab_")

    def encode(self, text: str) -> list[list[int]]:
        return [[self.char_to_id[ch] for ch in text]]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        size = len(_ALPHABET)
        logits = [0.0] * size
        if self._call_count < len(self.script):
            wanted = self.script[self._call_count]
            if wanted in self.char_to_id:
                logits[self.char_to_id[wanted]] = 100.0
        self._call_count += 1
        return logits

    def get_path_to_vocab_file(self) -> str:
        vocab = {encode_vocab_token(ch): i for ch, i in self.char_to_id.items()}
        path = Path(self._vocab_dir) / "vocab.json"
        path.write_text(json.dumps(vocab), encoding="utf-8")
        return str(path)
