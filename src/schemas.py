"""Pydantic models for the function-calling tool's inputs and outputs."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ParameterSpec(BaseModel):
    """Schema for a single function parameter.

    Extra keys found in the source JSON (e.g. a free-text ``description``)
    are kept but not interpreted, so the tool stays forward compatible with
    slightly richer function definitions.
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(description="JSON type: string, number, integer, "
                                  "boolean, array or object.")
    enum: list[Any] | None = Field(
        default=None,
        description="If set, the value must be exactly one of these "
                    "literals (e.g. a firmware field restricted to a "
                    "predefined set of options).",
    )
    items: "ParameterSpec | None" = Field(
        default=None, description="Element schema, when type == 'array'."
    )

    @field_validator("type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        """Normalize the type name to lower case.

        Unknown type names are not rejected here: the generator falls back
        to a safe string-like strategy for anything it does not recognize,
        so a slightly unusual schema still produces an answer instead of
        crashing the whole run.
        """
        return value.strip().lower()


class FunctionDefinition(BaseModel):
    """One entry of ``functions_definition.json``."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str = ""
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: dict[str, Any] | None = None


class PromptItem(BaseModel):
    """One entry of ``function_calling_tests.json``."""

    model_config = ConfigDict(extra="allow")

    prompt: str


class OutputEntry(BaseModel):
    """One entry written to the output JSON file."""

    prompt: str
    fn_name: str
    args: dict[str, Any]
