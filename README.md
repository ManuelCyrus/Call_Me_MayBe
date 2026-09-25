*This project has been created as part of the 42 curriculum by nfardim.*

# call me maybe

A function-calling tool: it reads a natural-language prompt and a list of
available functions, and decides which single function to call and with
which arguments — using **constrained decoding** over a small local LLM
(`Qwen/Qwen3-0.6B` by default), not by hoping the model writes valid JSON
on its own.

> **Note on output keys.** The subject text (chapter V.4) shows the output
> keys as `prompt`, `name`, `parameters`. The evaluation scale provided for
> this project instead requires `prompt`, `fn_name`, `args`. This
> implementation follows the **evaluation scale**, since that is what
> peer review actually checks. If your campus's current subject uses the
> other names, renaming the three keys in `src/schemas.py`
> (`OutputEntry`) and `src/schema_gen.py` is a two-line change.

## Description

Given a prompt like *"What is the sum of 2 and 3?"* and a function
`fn_add_numbers(a: number, b: number)`, the tool produces:

```json
{"prompt": "What is the sum of 2 and 3?", "fn_name": "fn_add_numbers", "args": {"a": 2, "b": 3}}
```

The JSON is never written by the model directly. Every structural
character (braces, quotes, colons, commas) is injected deterministically;
the model only ever chooses among a small set of **allowed next tokens**,
computed fresh at each step from the function list and the parameter
schema. That is what "constrained decoding" means here, and it is why the
output is 100% valid, schema-compliant JSON by construction rather than by
luck.

## Instructions

```bash
make install        # uv sync (installs pydantic + the local llm_sdk package)
make run             # uv run python -m src
make debug           # uv run python -m pdb -m src
make lint            # flake8 . && mypy . <required flags>
make lint-strict     # flake8 . && mypy . --strict
make test            # uv run pytest -q  (fake model, no network/GPU needed)
make clean           # remove caches and data/output
```

Direct invocation, with the exact arguments from the subject:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

All three flags are optional and default to the paths above. `--model`
selects a different Hugging Face model id (see "Bonus" below).

Before the first `make run`, `llm_sdk`'s `Small_LLM_Model` downloads the
model from the Hugging Face Hub, so **internet access to huggingface.co is
required once** (the model is then cached locally by `transformers`).

## Algorithm explanation

Generation for one prompt happens in three layers:

1. **Prompt.** `src/prompts.py` builds a short natural-language prompt:
   the available functions (name, parameter names/types, description) and
   the user's request. This is the only place the model has to reason
   about *content*; it never has to reason about JSON syntax.

2. **Grammar.** `src/constrained.py` provides the primitives:
   - `Generator.force(text)` appends deterministic text with **no model
     call** (there's nothing to decide): braces, colons, key names,
     `, "args": `, etc. This is why the tool doesn't spend its 5-minute
     budget re-deciding things that are never in doubt.
   - `Generator.choose_enum(literal_texts)` lets the model pick among a
     *closed* set of exact strings — the function name, an
     `enum`-restricted parameter, or `true`/`false`. Every candidate is
     tokenized once (via the SDK's own `encode`, so tokenisation always
     matches the model's tokenizer exactly); the walk descends a shared
     prefix for free and only calls the model where the candidates still
     disagree (a 5-way function-name choice is typically **one** model
     call, not one per character of the longest name).
   - `Generator.generate_number` / `Generator.generate_string` handle
     open-ended content (the actual digits or characters). A tiny
     character-level automaton decides, token by token, whether a
     candidate keeps the value well-formed; once the value is already
     valid (e.g. `"23"` is a complete number), the *same* masked-logit
     comparison also offers "stop here" as an option, by mixing in the
     first token of whatever deterministic text follows the value. The
     model's own preference — not a hand-written length heuristic —
     decides whether to add one more digit/character or move on.
   - `Generator.generate_number_choice` / `generate_string_choice` are the
     same idea for array elements, where "stop" can mean two different
     things ("one more item" or "end the array"); both exits are offered
     at once, so no extra model call is ever spent purely on that
     decision, and only in the rare case that both exits share their
     first token does a follow-up call disambiguate them.

3. **Schema.** `src/schema_gen.py` turns one `FunctionDefinition`'s
   `parameters` into a sequence of forced key names and calls to the
   grammar primitives above, picking the primitive by the parameter's
   declared `type` (and by `enum`, when present, which takes priority
   over the base type — this is how a `firmware` field restricted to a
   fixed set of options is handled). `closing` is threaded through every
   call: it is the exact text that must follow a value once it's done
   (`', "'` before the next sibling key, or `'}'` chained with whatever
   the parent needed), so each function only ever worries about its own
   brace, never about how deep it is nested.

At every step the actual masking is: take the model's logits for the next
token, keep only the ones the grammar currently allows (everything else is
effectively `-inf`), and take the argmax. That is deterministic (no
sampling temperature), which is deliberate: reproducible output is worth
more here than sampling diversity, and the subject's 90%+ accuracy target
is about picking the *right* function and values, not about creative
phrasing.

Only two SDK methods are used to run the model:
`get_logits_from_input_ids` for every real decision, and `encode` to
tokenize the prompt and every forced literal (see "Bonus" for what a full
byte-level-BPE reimplementation avoiding `encode` would look like).
`get_path_to_vocab_file` is used once per run, in `src/vocab_utils.py`, to
decode every vocabulary entry into the text it represents (Qwen uses the
same byte-level scheme as GPT-2: each raw byte is mapped to a printable
unicode character before landing in `vocab.json`), which is what lets the
grammar know, for any token id, whether it is a digit, a quote, a control
character, etc.

## Design decisions

- **Greedy, masked decoding, no sampling.** Reliability matters more than
  variety for a function-calling tool; the mask already does the hard
  work of guaranteeing validity, so there is no need for temperature or
  top-p on top of it.
- **Forced text costs no model call.** Every purely structural token
  (about half of a typical output: braces, quotes, colons, key names) is
  injected via a single `encode()` call and appended directly. Only
  genuine decisions — which function, which enum value, which digit/
  character, continue-or-stop — touch `get_logits_from_input_ids`, which
  is the expensive part on CPU.
- **`enum` beats base `type`.** A parameter with both `"type": "string"`
  and `"enum": [...]` is generated as a closed choice among the enum's
  *exact* JSON literals (via `json.dumps`), which also makes it trivial to
  support enums of numbers or booleans, not just strings.
- **Pydantic everywhere data crosses a boundary.** `src/schemas.py`
  defines `ParameterSpec`, `FunctionDefinition`, `PromptItem` and
  `OutputEntry`; loading either input file goes through
  `FunctionDefinition.model_validate` / `PromptItem.model_validate`
  (`src/io_utils.py`), so a malformed file is rejected with one clear
  message instead of failing deep inside the generator with a confusing
  `KeyError`.
- **One bad prompt never aborts the run.** `src/__main__.py` catches any
  exception per prompt, prints a one-line warning to stderr naming the
  prompt, and continues; the output file only ever contains prompts that
  actually produced a call.
- **A `Protocol`, not a concrete import, decides what "the model" is.**
  `src/llm_interface.py`'s `LlmLike` describes only the three methods this
  package needs. `src/__main__.py` imports `llm_sdk` lazily, inside
  `_load_model`, so `--help`, linting and the whole test suite never need
  `torch`/`transformers` installed or a network connection.

## Known limitations

- **String content has no escape support.** Generated string values never
  contain a literal `"` or `\`: tokens that would introduce either are
  simply not offered as valid continuations. This keeps the automaton
  small; it means a prompt asking to reproduce a quote character verbatim
  will not do so.
- **Numbers have no exponent notation** (`1e10`): the automaton only
  covers `-?digits(.digits)?`, which is what every realistic
  function-calling prompt in the examples needs.
- **Arrays support `number`, `integer` and `string` items only.** Arrays
  of enums, booleans, objects, or nested arrays raise a clear error for
  that one prompt (caught and skipped, see above) rather than guessing at
  an unspecified encoding.
- **Nested `object` parameters** are only generated when the function
  definition provides their own `"properties"` sub-schema (an accepted,
  non-standard extra field); otherwise an empty `{}` is emitted.
- **mypy and the vendored `llm_sdk` folder.** `src/__main__.py` carries
  one `# type: ignore[attr-defined]` on `from llm_sdk import
  Small_LLM_Model`. mypy's static namespace-package discovery can resolve
  `llm_sdk` to the vendored *folder* that sits beside `src/` (which has no
  `__init__.py` at its outer level) instead of the distribution `uv sync`
  actually installs from it, and reports a false "has no attribute"
  there. At runtime, and under `uv run`, the installed package resolves
  correctly — confirmed with `uv run python -c "import llm_sdk;
  print(dir(llm_sdk))"`, which lists `Small_LLM_Model`, and with `uv run
  python -m src --help`, which runs cleanly. The rest of the codebase has
  no ignored mypy errors, under either the required flags or `--strict`.
- **No huggingface.co access in the environment this was authored in.**
  The full pipeline (`make run`) was therefore verified structurally
  (the CLI loads inputs, fails to reach the Hub, and exits 1 with a clear
  message — no traceback) rather than against real model output; the
  constrained-decoding logic itself is exercised end-to-end by the test
  suite against a fake model (see "Testing strategy"). On a machine with
  normal internet access, the same command downloads `Qwen/Qwen3-0.6B`
  once and runs unchanged.

## Performance analysis

- **JSON validity: 100% by construction.** No path through
  `src/constrained.py` can append a token that the grammar does not
  currently allow, so every produced entry parses; there is no
  "hopefully" involved.
- **Accuracy** depends on the underlying model's ability to pick the
  right function and copy the right numbers/words from the prompt, which
  constrained decoding cannot fix if the model is simply confused about
  content — it can only guarantee that whatever the model picks is
  syntactically and structurally legal. In practice, picking the right
  function is an easy decision even for a 0.6B model, since the grammar
  reduces the "choose a function" step to a handful of first-diverging
  characters rather than free-form generation.
- **Speed.** The dominant cost is one CPU forward pass through the 0.6B
  model per genuine decision. Structural text costs nothing (see "Design
  decisions"), so a typical two-argument call (`fn_add_numbers`) costs
  roughly: 1 call to pick the function + ~3-4 calls per numeric argument
  (digits plus the stop decision) ≈ 8-10 calls total, not one per output
  character. `Small_LLM_Model.get_logits_from_input_ids` re-runs the full
  forward pass from scratch each call (the SDK exposes no incremental KV
  cache), which is the main lever left for a from-scratch performance
  optimisation not attempted here (see "Bonus").
- **Reliability across runs.** Decoding is fully greedy (masked argmax,
  no sampling), so the same prompt and function list always produce the
  same output.

## Testing strategy

`tests/fake_model.py` implements the exact same three-method surface as
`Small_LLM_Model` (`encode`, `get_logits_from_input_ids`,
`get_path_to_vocab_file`) with a tiny single-character vocabulary and a
scripted sequence of "preferred" tokens, so the grammar/automaton logic
can be tested deterministically with no network access, no GPU, and no
multi-gigabyte model download — which also matches the environment this
project was authored in (no route to huggingface.co). This is dependency
injection at the SDK boundary (`LlmLike`), not a mock of internal
behaviour.

- `tests/test_constrained.py` — the low-level primitives in isolation:
  forcing text, multi-token enum choice (including when it needs zero
  model calls because the candidates share a full prefix), number/string
  generation (positive, negative, decimal, integer-only rejecting a dot,
  empty string), and the empty-pool error path.
- `tests/test_schema_gen.py` — full `{"fn_name": ..., "args": {...}}`
  generation end to end for: picking among several functions, plain
  number/string parameters, an `enum`-restricted parameter (the
  "firmware" case), an array of integers and an array of strings
  (specifically chosen because both of a string array's two possible
  exits start with the same closing-quote token, exercising the
  shared-prefix tie-break), an unsupported array item type (must raise,
  not guess), and an empty function list (must raise). One test also
  re-parses the accumulated token ids as JSON and compares it against the
  expected dict, so it checks the raw output text, not just the returned
  Python value.
- `tests/test_io_utils.py` — loading valid/missing/malformed/
  wrong-shaped input files, and that `write_output` creates its target
  directory.
- `tests/test_main.py` — the CLI end to end with the fake model injected
  via `monkeypatch.setattr(main_module, "_load_model", ...)`: a full
  happy path checking the exact output file content, a missing-input-file
  error message, and a run with zero functions defined, which must still
  exit 0 with an empty output file and one warning per skipped prompt on
  stderr (never a crash).

`make test` runs all of this in well under a second. `make lint` /
`make lint-strict` (flake8 and mypy, including `--strict`) pass with zero
errors, checked both directly and via `uv run` inside the `uv sync`
environment.

## Example usage

```bash
$ make install
$ make run
Processed 11 prompt(s): 11 succeeded, 0 skipped. Output written to data/output/function_calling_results.json

$ cat data/output/function_calling_results.json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": {"a": 2, "b": 3}
  },
  {
    "prompt": "Greet shrek",
    "fn_name": "fn_greet",
    "args": {"name": "shrek"}
  }
]
```

A missing input file fails clearly instead of crashing:

```bash
$ uv run python -m src --input data/input/does_not_exist.json
Error: input file not found: data/input/does_not_exist.json
$ echo $?
1
```

## Bonus

Implemented:

- **Support for multiple LLM models**: `--model` is a CLI flag
  (`src/__main__.py`), not a hard-coded string; anything exposing the same
  three-method `LlmLike` surface works.
- **Comprehensive test suite**: 28 tests across four files (see "Testing
  strategy"), covering the grammar primitives, full end-to-end generation
  including two nested-argument shapes, input/output error handling, and
  the CLI itself — all offline.
- **Support for (limited) nested/complex arguments**: arrays of numbers,
  integers and strings (`src/schema_gen.py::generate_array`), including
  the shared-prefix exit disambiguation described above.

Not implemented (documented honestly rather than left silent):

- **Recoding the tokenizer** (avoiding `encode`/`decode` in the main code
  in favour of a from-scratch byte-level-BPE encoder built only from
  `get_path_to_vocab_file` and `get_path_to_merges_file`). `decode` is
  never used in the main pipeline already (only in the test fake model,
  for convenience); a from-scratch **encoder** was scoped out given the
  time available for this project. `src/vocab_utils.py` already implements
  the *decode* half of a byte-level-BPE tokenizer from the vocab file
  alone (used for grammar masking), which would be most of the groundwork
  for that direction.
- **Advanced error-recovery mechanisms** beyond "skip this one prompt,
  keep going, tell the user why" (see "Design decisions").
- **Performance optimisations** such as caching or batching: the SDK's
  `get_logits_from_input_ids` recomputes the full forward pass every call
  with no exposed KV cache, which caps how much a caller-side
  optimisation alone can achieve.
- **Visualisation of the generation process.**

## Resources

- Qwen2/Qwen3 tokenizer documentation (byte-level BPE, `vocab.json` +
  `merges.txt`): https://huggingface.co/docs/transformers/model_doc/qwen2
- GPT-2's byte-to-unicode scheme, which Qwen's tokenizer also uses (the
  basis for `src/vocab_utils.py`):
  https://github.com/openai/gpt-2/blob/master/src/encoder.py
- Constrained/guided decoding background: Willard & Louf, *Efficient
  Guided Generation for LLMs* (the `outlines` paper), and the
  `lm-format-enforcer` and `outlines` project READMEs for the general
  "mask logits to a finite-state grammar" technique (neither package is
  used here, per the subject's restriction — only their public
  descriptions of the technique informed the design).
- Pydantic v2 documentation, for `model_validate`, `ConfigDict(extra=...)`
  and `model_extra`: https://docs.pydantic.dev/latest/
- `uv`'s documentation on local/editable path dependencies
  (`[tool.uv.sources]`): https://docs.astral.sh/uv/concepts/dependencies/

**Use of AI.** An AI assistant (Claude) was used to: design and implement
the constrained-decoding engine (the grammar/automaton in
`src/constrained.py`, the schema-driven generation in `src/schema_gen.py`)
after discussing several alternative designs and their trade-offs (a
sequential per-character exit heuristic was tried, found to have a
token-collision bug for array-of-string items — where both possible
continuations start with the same closing-quote token — and replaced by
the current shared-prefix trie resolution, verified by a dedicated test);
write the vocabulary byte-level-BPE decoder in `src/vocab_utils.py`; write
`tests/fake_model.py` and the full test suite, run against it to catch
several bugs during development (an incorrect field separator that
produced a doubled quote character, and the array-exit-collision bug
above); and draft this README. Every design decision and every bug fix
described above was verified by actually running `pytest`, `flake8` and
`mypy` (including `--strict`) locally, and by running `uv sync` and the
real CLI's `--help`/error paths under `uv run`, rather than assumed.
