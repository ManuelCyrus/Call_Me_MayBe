"""Helpers to turn a token id into the literal text it represents.

Qwen (like GPT-2) uses a byte-level BPE tokenizer: every raw byte is first
mapped to a printable unicode character, and ``vocab.json`` maps those
"byte strings" to integer token ids. To reason about what characters a
token id will actually add to the output (so we can decide whether it is
allowed at a given point in the JSON grammar) we need the inverse of that
byte-to-unicode mapping.
"""
from __future__ import annotations

import json
from functools import lru_cache


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    """Build GPT-2's byte-to-unicode table (public, well known scheme).

    Returns
    -------
    dict[int, str]
        Maps a raw byte value (0-255) to the single unicode character
        used to represent it inside a byte-level BPE vocabulary file.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _unicode_to_bytes() -> dict[str, int]:
    """Invert :func:`_bytes_to_unicode`."""
    return {v: k for k, v in _bytes_to_unicode().items()}


def decode_vocab_token(token: str) -> str:
    """Decode one byte-level-BPE vocabulary entry into real text.

    Parameters
    ----------
    token : str
        A key from ``vocab.json`` (e.g. ``"Ġhello"``).

    Returns
    -------
    str
        The text this token contributes to the generated string (e.g.
        ``" hello"``). Bytes that do not form valid UTF-8 on their own
        (common for tokens that are only half of a multi-byte character)
        are replaced rather than raising, since we only use this text to
        check which *characters* a token would add.
    """
    table = _unicode_to_bytes()
    raw = bytes(table.get(ch, ord(ch)) & 0xFF for ch in token)
    return raw.decode("utf-8", errors="replace")


def encode_vocab_token(text: str) -> str:
    """Inverse of :func:`decode_vocab_token`.

    Turns literal text into the byte-level-BPE key it would appear as in
    ``vocab.json``. Mainly used by the test suite to build a small,
    self-contained fake vocabulary that the same decoding logic can read.
    """
    table = _bytes_to_unicode()
    return "".join(table[b] for b in text.encode("utf-8"))


def load_vocab_texts(vocab_path: str) -> dict[int, str]:
    """Load ``vocab.json`` and decode every entry to its literal text.

    Parameters
    ----------
    vocab_path : str
        Path returned by ``Small_LLM_Model.get_path_to_vocab_file()``.

    Returns
    -------
    dict[int, str]
        Maps token id -> the text it represents.
    """
    with open(vocab_path, "r", encoding="utf-8") as fh:
        raw_vocab: dict[str, int] = json.load(fh)
    return {tid: decode_vocab_token(tok) for tok, tid in raw_vocab.items()}
