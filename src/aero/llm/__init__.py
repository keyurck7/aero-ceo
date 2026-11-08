"""Every LLM call in AERO-CEO goes through this package (ADR 0008).

* ``client``: an OpenAI-compatible chat client (vLLM serves Qwen this way) with JSON-schema structured outputs,
  retries and timeouts.
* ``prompts``: versioned prompt templates (``prompt_id@version`` is stored next to every output).
* ``structured``: render a prompt, request schema-constrained JSON, validate it with Pydantic, repair once.
* ``cassette``: record/replay, so CI and evals are deterministic and reviewable.
* ``fake``: a scripted LLM for unit tests.
"""

from aero.llm.client import (
    LLM,
    Completion,
    LLMError,
    LLMRequest,
    LLMUnavailableError,
    OpenAICompatibleLLM,
    build_llm,
)
from aero.llm.prompts import Prompt, PromptRegistry, default_registry
from aero.llm.structured import StructuredOutputError, generate

__all__ = [
    "LLM",
    "Completion",
    "LLMError",
    "LLMRequest",
    "LLMUnavailableError",
    "OpenAICompatibleLLM",
    "Prompt",
    "PromptRegistry",
    "StructuredOutputError",
    "build_llm",
    "default_registry",
    "generate",
]
