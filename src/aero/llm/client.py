"""OpenAI-compatible chat client for a vLLM-served model, plus the ``LLM`` protocol everything else uses."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import requests

from aero.log import get_logger

if TYPE_CHECKING:
    from aero.config import Settings

log = get_logger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


class LLMUnavailableError(LLMError):
    """No model endpoint is configured (or reachable). Callers surface this; they never fall back to heuristics."""


@dataclass(frozen=True)
class LLMRequest:
    prompt_id: str
    prompt_version: str
    messages: list[dict[str, str]]
    model: str
    schema_name: str | None = None
    schema: dict[str, Any] | None = None
    temperature: float = 0.0
    top_p: float | None = None
    max_tokens: int = 2048
    seed: int | None = None

    def cache_payload(self) -> dict[str, Any]:
        """Everything that determines the output (the cassette key)."""
        return {
            "prompt": f"{self.prompt_id}@{self.prompt_version}", "model": self.model, "messages": self.messages,
            "schema": self.schema, "temperature": self.temperature, "top_p": self.top_p,
            "max_tokens": self.max_tokens, "seed": self.seed,
        }


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    cached: bool = False


class LLM(Protocol):
    model: str

    def complete(self, request: LLMRequest) -> Completion: ...


class OpenAICompatibleLLM:
    """POST /chat/completions with ``response_format: json_schema`` (vLLM structured outputs)."""

    def __init__(self, base_url: str, model: str, *, api_key: str = "", timeout: float = 180.0,
                 max_retries: int = 2, enable_thinking: bool = False,
                 session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.session = session or requests.Session()
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def body(self, request: LLMRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model, "messages": request.messages, "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            # Qwen3 chat template switch; other models ignore unknown template kwargs.
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.seed is not None:
            body["seed"] = request.seed
        if request.schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": request.schema_name or "output", "schema": request.schema, "strict": True}}
        return body

    def complete(self, request: LLMRequest) -> Completion:
        url = f"{self.base_url}/chat/completions"
        body = self.body(request)
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(url, data=json.dumps(body), headers=self.headers, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == self.max_retries:
                    raise LLMUnavailableError(f"LLM endpoint {self.base_url} unreachable: {exc}") from exc
                log.warning("llm retry", extra={"attempt": attempt + 1, "error": str(exc)})
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]
                    return Completion(text=choice["message"].get("content") or "", model=data.get("model", self.model),
                                      usage={k: int(v) for k, v in (data.get("usage") or {}).items()
                                             if isinstance(v, int)},
                                      finish_reason=choice.get("finish_reason"))
                if resp.status_code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    raise LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
                log.warning("llm retry", extra={"attempt": attempt + 1, "status": resp.status_code})
            time.sleep(delay)
            delay *= 2
        raise LLMError("unreachable")  # pragma: no cover


class _Unconfigured:
    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, request: LLMRequest) -> Completion:
        raise LLMUnavailableError(
            "No LLM endpoint configured. Set AERO_LLM_BASE_URL (e.g. http://localhost:8000/v1 for vLLM), or "
            "AERO_LLM_CASSETTE_MODE=replay to use recorded outputs. Signals, verification and partner ranking "
            "need a model; there is deliberately no heuristic fallback (ADR 0008)."
        )


def build_llm(settings: Settings, *, judge: bool = False) -> LLM:
    """The configured model, wrapped in the cassette recorder/replayer when one is enabled."""
    from aero.llm.cassette import CassetteLLM

    model = settings.judge_model_name if judge else settings.llm_model
    inner: LLM
    if settings.llm_base_url:
        inner = OpenAICompatibleLLM(settings.llm_base_url, model, api_key=settings.llm_api_key,
                                    timeout=settings.llm_timeout_seconds, max_retries=settings.llm_max_retries,
                                    enable_thinking=settings.llm_enable_thinking)
    else:
        inner = _Unconfigured(model)
    if settings.llm_cassette_mode == "off":
        return inner
    return CassetteLLM(inner, settings.llm_cassette_dir, settings.llm_cassette_mode)
