"""Normalize the SDK's ``encode()`` output into a plain list of ints.

``Small_LLM_Model.encode`` returns a 2-D ``torch.Tensor`` (shape
``[1, seq_len]``); the fake model used in tests returns a plain
``list[list[int]]`` for the same shape, so no test needs torch installed.
This module supports both without importing torch itself, keeping the
rest of the package torch-free.
"""
from __future__ import annotations


def to_id_list(encoded: object) -> list[int]:
    """Flatten a batch-of-one tensor-or-list of token ids to ``list[int]``.

    Parameters
    ----------
    encoded : object
        Whatever ``model.encode(text)`` returned.

    Returns
    -------
    list[int]
        The token ids as plain Python integers.

    Raises
    ------
    TypeError
        If ``encoded`` is not a tensor-like or list-like object.
    """
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, (list, tuple)):
        seq: list[object] = list(encoded)
        if seq and isinstance(seq[0], (list, tuple)):
            seq = list(seq[0])
        return [int(x) for x in seq if isinstance(x, (int, float, str))]
    raise TypeError(f"unexpected encode() output type: {type(encoded)!r}")
