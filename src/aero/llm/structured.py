"""Schema-constrained generation: render a versioned prompt, request JSON matching a Pydantic model, validate.

The server constrains decoding to the JSON schema (vLLM structured outputs), and the response is validated
again here because a constrained grammar cannot express every rule (e.g. cross-field checks). One repair
round is allowed: the model sees its own output and the validation errors. After that the call fails loudly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from aero.llm.client import LLM, Completion, LLMError, LLMRequest
from aero.llm.prompts import Prompt

T = TypeVar("T", bound=BaseModel)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class StructuredOutputError(LLMError):
    def __init__(self, prompt: Prompt, text: str, error: str) -> None:
        super().__init__(f"{prompt.ref}: output failed validation after repair: {error[:500]}")
        self.prompt_ref = prompt.ref
        self.text = text


def clean_json_text(text: str) -> str:
    return _FENCE.sub("", _THINK.sub("", text).strip()).strip()


def generate(
    llm: LLM,
    prompt: Prompt,
    output_model: type[T],
    variables: Mapping[str, object],
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    max_tokens: int | None = None,
) -> tuple[T, Completion]:
    messages = prompt.render(variables)
    request = LLMRequest(
        prompt_id=prompt.id, prompt_version=prompt.version, messages=messages, model=llm.model,
        schema_name=output_model.__name__, schema=output_model.model_json_schema(),
        temperature=prompt.temperature if temperature is None else temperature, top_p=top_p,
        max_tokens=max_tokens or prompt.max_tokens, seed=seed,
    )
    completion = llm.complete(request)
    try:
        return output_model.model_validate_json(clean_json_text(completion.text)), completion
    except ValidationError as exc:
        error = exc.json(include_url=False)
    repair = LLMRequest(
        prompt_id=prompt.id, prompt_version=prompt.version, model=llm.model, schema_name=request.schema_name,
        schema=request.schema, temperature=0.0, max_tokens=request.max_tokens, seed=seed,
        messages=[*messages, {"role": "assistant", "content": completion.text},
                  {"role": "user", "content": "Your previous output did not validate against the required JSON "
                                              f"schema. Errors: {error[:2000]}\nReturn only the corrected JSON."}],
    )
    fixed = llm.complete(repair)
    try:
        return output_model.model_validate_json(clean_json_text(fixed.text)), fixed
    except ValidationError as exc:
        raise StructuredOutputError(prompt, fixed.text, exc.json(include_url=False)) from exc
