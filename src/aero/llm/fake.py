"""A scripted LLM for unit tests: a handler maps each request to a response (dict -> JSON, or raw text)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from aero.llm.client import Completion, LLMRequest

Handler = Callable[[LLMRequest], "dict[str, Any] | str"]


class FakeLLM:
    def __init__(self, handler: Handler, model: str = "fake-model") -> None:
        self.handler = handler
        self.model = model
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> Completion:
        self.calls.append(request)
        out = self.handler(request)
        return Completion(text=out if isinstance(out, str) else json.dumps(out), model=self.model)

    def calls_for(self, prompt_id: str) -> list[LLMRequest]:
        return [c for c in self.calls if c.prompt_id == prompt_id]


def variables_of(request: LLMRequest) -> str:
    """The rendered user message, for handlers that route on content."""
    return next((m["content"] for m in reversed(request.messages) if m["role"] == "user"), "")
