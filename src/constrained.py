"""Constrained decoding: force the model's output to be valid JSON that
matches a given schema, by masking invalid tokens to -inf before picking
the next token (never by hoping the prompt alone produces good JSON).

The grammar is generated incrementally rather than pre-compiled into one
big automaton, because the shape of ``args`` depends on *which* function
was chosen, which is itself a decision the model makes along the way.
Three building blocks cover the whole grammar:

* :func:`encode_ids` / forced literals -- structural text with no
  ambiguity (braces, colons, key names, quotes) is injected directly: the
  tokenizer is deterministic, so there is nothing to decide.
* :class:`EnumChoice` -- a closed set of literal values (the function
  name, an ``enum``-restricted parameter, or a boolean). The model picks
  among their token sequences one branch point at a time.
* :func:`generate_number` / :func:`generate_string` -- open-ended content
  (the actual digits or characters). A tiny character-level automaton
  decides, per token, whether it is still a valid continuation; the model
  also gets the option to stop as soon as the value is already valid.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from src.ids_utils import to_id_list
from src.llm_interface import LlmLike
from src.vocab_utils import load_vocab_texts

GetLogits = Callable[[list[int]], list[float]]

MAX_VALUE_TOKENS = 40
MAX_ARRAY_ITEMS = 8

_STRING_FORBIDDEN = re.compile(r'["\\\x00-\x1f]')
_NUMBER_CHARS = re.compile(r"^[0-9.\-]+$")


class GenerationError(RuntimeError):
    """Raised when the model cannot be advanced (e.g. no valid token)."""


@dataclass
class VocabIndex:
    """Token id -> literal text, plus a couple of precomputed subsets.

    Building these subsets is a single O(vocab size) pass done once per
    run (not per generated token), so it has no meaningful impact on the
    5-minute performance budget.
    """

    id_to_text: dict[int, str]
    string_content_ids: frozenset[int] = field(init=False)
    number_char_ids: frozenset[int] = field(init=False)

    def __post_init__(self) -> None:
        string_ids = []
        number_ids = []
        for tid, text in self.id_to_text.items():
            if text == "":
                continue
            if not _STRING_FORBIDDEN.search(text):
                string_ids.append(tid)
            if _NUMBER_CHARS.match(text):
                number_ids.append(tid)
        self.string_content_ids = frozenset(string_ids)
        self.number_char_ids = frozenset(number_ids)

    @classmethod
    def from_model(cls, model: LlmLike) -> "VocabIndex":
        """Build the index from the model's own vocabulary file."""
        vocab_path = model.get_path_to_vocab_file()
        return cls(load_vocab_texts(vocab_path))


def _argmax(logits: list[float], allowed: dict[int, float]) -> int:
    """Return the allowed id with the highest logit.

    Parameters
    ----------
    logits : list[float]
        Full logits vector for the next token, as returned by the model.
    allowed : dict[int, float]
        Maps each allowed token id to a tie-break bonus (0.0 for plain
        candidates; used to prefer, at equal logit, the token that would
        end a value once it is already valid -- a harmless, deterministic
        tie-break that never overrides a genuine model preference).
    """
    if not allowed:
        raise GenerationError("no valid token at this step of the grammar")
    best_id = -1
    best_score = float("-inf")
    for tid, bonus in allowed.items():
        score = logits[tid] + bonus
        if score > best_score:
            best_score = score
            best_id = tid
    return best_id


class Generator:
    """Runs one constrained-decoding session against a live model."""

    def __init__(self, model: LlmLike, vocab: VocabIndex,
                 input_ids: list[int]) -> None:
        self.model = model
        self.vocab = vocab
        self.ids: list[int] = list(input_ids)

    def encode_ids(self, text: str) -> list[int]:
        """Tokenize ``text`` the same way the model's own prompt was."""
        return to_id_list(self.model.encode(text))

    def force(self, text: str) -> None:
        """Append deterministic, unambiguous text with no model call."""
        self.ids.extend(self.encode_ids(text))

    def next_logits(self) -> list[float]:
        return self.model.get_logits_from_input_ids(self.ids)

    def _trie_choose(self, token_seqs: list[list[int]]) -> int:
        """Walk several token sequences at once, appending the shared
        prefix for free and calling the model only where they diverge.

        Returns the index into ``token_seqs`` of the sequence that was
        fully matched; every token of it has already been appended to
        ``self.ids``.
        """
        depth = 0
        remaining = list(range(len(token_seqs)))
        while True:
            still_open = [i for i in remaining if len(token_seqs[i]) > depth]
            if not still_open:
                break
            next_ids = {token_seqs[i][depth] for i in still_open}
            if len(next_ids) == 1:
                chosen_id = next(iter(next_ids))
            else:
                logits = self.next_logits()
                allowed = {tid: 0.0 for tid in next_ids}
                chosen_id = _argmax(logits, allowed)
            self.ids.append(chosen_id)
            remaining = [
                i for i in still_open if token_seqs[i][depth] == chosen_id
            ]
            depth += 1
            if len(remaining) == 1 and len(token_seqs[remaining[0]]) == depth:
                break
        return remaining[0]

    def choose_enum(self, literal_texts: list[str]) -> str:
        """Let the model pick one of several exact literal strings.

        Each candidate is tokenized once; the choice is made one token at
        a time only where the candidates actually still disagree, so a
        function-name decision between five options usually costs a
        single model call, not one call per token of the longest name.
        """
        candidates = [self.encode_ids(t) for t in literal_texts]
        return literal_texts[self._trie_choose(candidates)]

    def _generate_value(
        self,
        allowed_ids: frozenset[int],
        is_accepting: Callable[[str], bool],
        step: Callable[[str, str], str | None],
        exit_options: list[list[int]],
    ) -> tuple[str, int]:
        """Shared loop for open-ended values (numbers and strings).

        At each step the pool of legal next tokens is the union of the
        content tokens that keep the value well-formed *and*, once the
        value is already a complete, valid piece of JSON, the first token
        of each deterministic text in ``exit_options`` that may follow it.
        Both come from the same masked-logit comparison: the model
        decides, token by token, whether to keep extending the value or
        to close it -- and, when more than one exit is offered (used for
        array items, where "close" can mean "one more item" or "end the
        array"), *which* closing applies.

        Returns
        -------
        tuple[str, int]
            The generated content, and the index into ``exit_options``
            that was chosen (its tokens have already been appended).
        """
        text = ""
        for _ in range(MAX_VALUE_TOKENS):
            accepting = is_accepting(text)
            pool: dict[int, float] = {}
            candidate_next_state: dict[int, str] = {}
            for tid in allowed_ids:
                new_text = step(text, self.vocab.id_to_text.get(tid, ""))
                if new_text is not None:
                    pool[tid] = 0.0
                    candidate_next_state[tid] = new_text
            exit_first_ids: set[int] = set()
            if accepting:
                for ids in exit_options:
                    if ids:
                        exit_first_ids.add(ids[0])
                        pool.setdefault(ids[0], 0.05)
            if not pool:
                if accepting and exit_options:
                    return text, self._trie_choose(exit_options)
                raise GenerationError("stuck generating a value")
            logits = self.next_logits()
            chosen = _argmax(logits, pool)
            if chosen in exit_first_ids:
                # One or more exit options start with this exact token
                # (e.g. both "end the array" and "one more item" start by
                # closing the same string). Resolve which one via the same
                # shared-prefix trie walk used for enum choices -- this
                # only costs an extra model call in the rare case where
                # they are still tied beyond the first token.
                tied = [i for i, ids in enumerate(exit_options)
                        if ids and ids[0] == chosen]
                sub_idx = self._trie_choose([exit_options[i] for i in tied])
                return text, tied[sub_idx]
            self.ids.append(chosen)
            text = candidate_next_state[chosen]
        if exit_options:
            return text, self._trie_choose(exit_options)
        raise GenerationError("value exceeded the maximum length without closing")

    def _number_predicates(
        self, integer_only: bool
    ) -> tuple[Callable[[str], bool], Callable[[str, str], str | None]]:
        def is_accepting(text: str) -> bool:
            return bool(re.fullmatch(r"-?[0-9]+(\.[0-9]+)?", text)) if text else False

        def step(text: str, add: str) -> str | None:
            candidate = text + add
            pattern = r"-?[0-9]*" if integer_only else r"-?[0-9]*(\.[0-9]*)?"
            if candidate == "-" or (
                re.fullmatch(pattern, candidate) and candidate != ""
            ):
                return candidate
            return None

        return is_accepting, step

    @staticmethod
    def _string_predicates() -> tuple[
        Callable[[str], bool], Callable[[str, str], str | None]
    ]:
        def is_accepting(_text: str) -> bool:
            return True

        def step(text: str, add: str) -> str | None:
            return text + add

        return is_accepting, step

    def generate_number(self, exit_text: str, integer_only: bool) -> float | int:
        """Generate ``-?digits(.digits)?`` and return it as float/int."""
        is_accepting, step = self._number_predicates(integer_only)
        exit_ids = self.encode_ids(exit_text)
        text, _ = self._generate_value(
            self.vocab.number_char_ids, is_accepting, step, [exit_ids]
        )
        if not re.fullmatch(r"-?[0-9]+(\.[0-9]+)?", text):
            raise GenerationError(f"model produced an invalid number: {text!r}")
        return int(text) if integer_only or "." not in text else float(text)

    def generate_string(self, exit_text: str) -> str:
        """Generate free-form string content (already-open quote assumed).

        Backslash escapes are intentionally not offered as a valid
        continuation (see the README's "Known limitations" section): this
        keeps the automaton simple at the cost of not being able to emit
        a literal quote or backslash inside a generated string.
        """
        is_accepting, step = self._string_predicates()
        exit_ids = self.encode_ids(exit_text)
        text, _ = self._generate_value(
            self.vocab.string_content_ids, is_accepting, step, [exit_ids]
        )
        return text

    def generate_number_choice(
        self, more_text: str, end_text: str, integer_only: bool
    ) -> tuple[float | int, bool]:
        """Like :meth:`generate_number`, but for an array element: the same
        model decision also settles whether the array continues or ends.

        Returns
        -------
        tuple[float | int, bool]
            The value, and whether ``end_text`` (rather than ``more_text``)
            was the exit chosen -- i.e. whether the array is now closed.
        """
        is_accepting, step = self._number_predicates(integer_only)
        more_ids, end_ids = self.encode_ids(more_text), self.encode_ids(end_text)
        text, idx = self._generate_value(
            self.vocab.number_char_ids, is_accepting, step, [more_ids, end_ids]
        )
        if not re.fullmatch(r"-?[0-9]+(\.[0-9]+)?", text):
            raise GenerationError(f"model produced an invalid number: {text!r}")
        value = int(text) if integer_only or "." not in text else float(text)
        return value, idx == 1

    def generate_string_choice(
        self, more_text: str, end_text: str
    ) -> tuple[str, bool]:
        """Like :meth:`generate_string`, but for an array element."""
        is_accepting, step = self._string_predicates()
        more_ids, end_ids = self.encode_ids(more_text), self.encode_ids(end_text)
        text, idx = self._generate_value(
            self.vocab.string_content_ids, is_accepting, step, [more_ids, end_ids]
        )
        return text, idx == 1
