"""Record/replay for LLM calls ("VCR cassettes").

One JSON file per request, named by the SHA-256 of everything that determines the output, under
``<dir>/<prompt_id>@<version>/``. Recorded on Data Lab against the real model, committed, replayed in CI:
evals become deterministic and every model output used in a decision is reviewable in git.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from aero.llm.client import LLM, Completion, LLMRequest, LLMUnavailableError

CassetteMode = Literal["record", "replay", "record_missing"]


def request_key(request: LLMRequest) -> str:
    blob = json.dumps(request.cache_payload(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CassetteLLM:
    def __init__(self, inner: LLM, directory: str | Path, mode: CassetteMode) -> None:
        self.inner = inner
        self.model = inner.model
        self.directory = Path(directory)
        self.mode = mode

    def path_for(self, request: LLMRequest) -> Path:
        return self.directory / f"{request.prompt_id}@{request.prompt_version}" / f"{request_key(request)}.json"

    def complete(self, request: LLMRequest) -> Completion:
        path = self.path_for(request)
        if self.mode != "record" and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return Completion(text=data["text"], model=data["model"], usage=data.get("usage", {}),
                              finish_reason=data.get("finish_reason"), cached=True)
        if self.mode == "replay":
            raise LLMUnavailableError(f"no cassette for {request.prompt_id}@{request.prompt_version} "
                                      f"({path.name}); record it on Data Lab with AERO_LLM_CASSETTE_MODE=record")
        completion = self.inner.complete(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"request": request.cache_payload(), "text": completion.text,
                                    "model": completion.model, "usage": completion.usage,
                                    "finish_reason": completion.finish_reason},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        return completion
