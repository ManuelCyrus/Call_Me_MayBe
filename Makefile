# Makefile - call me maybe (function calling via constrained decoding)
#
# Mandatory rules per the subject: install, run, debug, clean, lint,
# lint-strict.

.PHONY: install run debug clean lint lint-strict test

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache
	find . -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	rm -rf data/output

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

# Not required by the subject, but useful during development: the unit
# tests run against a fake in-repo model (see tests/fake_model.py) so they
# need no network access, no GPU, and only a few hundred milliseconds.
test:
	uv run pytest -q
