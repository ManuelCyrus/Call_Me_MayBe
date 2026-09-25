"""Top-level orchestration: turn one prompt into one function call."""
from __future__ import annotations

from src.constrained import Generator, VocabIndex
from src.ids_utils import to_id_list
from src.llm_interface import LlmLike
from src.prompts import build_prompt
from src.schema_gen import generate_function_call
from src.schemas import FunctionDefinition, OutputEntry


class FunctionCaller:
    """Bundles a model, its decoded vocabulary and the available functions.

    The vocabulary is only decoded once per run (see
    :meth:`VocabIndex.from_model`), no matter how many prompts follow.
    """

    def __init__(self, model: LlmLike, functions: list[FunctionDefinition]) -> None:
        self.model = model
        self.functions = functions
        self.vocab = VocabIndex.from_model(model)

    def call(self, prompt: str) -> OutputEntry:
        """Run constrained decoding for a single prompt.

        Raises
        ------
        Exception
            Any failure while talking to the model or while generating a
            valid call (for example an empty function list) propagates to
            the caller, which decides whether to skip this one prompt or
            abort the whole run.
        """
        prompt_text = build_prompt(prompt, self.functions)
        input_ids = self._encode(prompt_text)
        gen = Generator(self.model, self.vocab, input_ids)
        fn_name, args = generate_function_call(gen, self.functions)
        return OutputEntry(prompt=prompt, fn_name=fn_name, args=args)

    def _encode(self, text: str) -> list[int]:
        return to_id_list(self.model.encode(text))
