"""Versioned prompt templates.

Prompts are reviewed artefacts: YAML files in ``aero/llm/prompts/`` with an ``id`` and a ``version``. Changing the
wording means bumping the version, and ``prompt_id@version`` is stored with every output, so any signal, verdict
or ranking can be traced to the exact instructions that produced it. Placeholders are ``{{name}}``; a missing or
unknown variable is an error, never a silent blank.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import yaml

_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")


@dataclass(frozen=True)
class Prompt:
    id: str
    version: str
    description: str
    system: str
    user: str
    temperature: float = 0.0
    max_tokens: int = 2048

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    @property
    def variables(self) -> set[str]:
        return set(_PLACEHOLDER.findall(self.system)) | set(_PLACEHOLDER.findall(self.user))

    def render(self, variables: Mapping[str, object]) -> list[dict[str, str]]:
        missing = self.variables - set(variables)
        extra = set(variables) - self.variables
        if missing or extra:
            raise ValueError(f"prompt {self.ref}: missing variables {sorted(missing)}, unknown {sorted(extra)}")

        def fill(template: str) -> str:
            return _PLACEHOLDER.sub(lambda m: str(variables[m.group(1)]), template)

        return [{"role": "system", "content": fill(self.system).strip()},
                {"role": "user", "content": fill(self.user).strip()}]


class PromptRegistry:
    def __init__(self, prompts: list[Prompt]) -> None:
        self._by_id: dict[str, Prompt] = {}
        for p in prompts:
            if p.id in self._by_id:
                raise ValueError(f"duplicate prompt id {p.id}")
            self._by_id[p.id] = p

    @classmethod
    def from_directory(cls, directory: str | Path) -> PromptRegistry:
        return cls([_load(Path(f).read_text(encoding="utf-8")) for f in sorted(Path(directory).glob("*.yaml"))])

    def get(self, prompt_id: str) -> Prompt:
        try:
            return self._by_id[prompt_id]
        except KeyError as exc:
            raise KeyError(f"unknown prompt {prompt_id!r}; known: {sorted(self._by_id)}") from exc

    def ids(self) -> list[str]:
        return sorted(self._by_id)


def _load(text: str) -> Prompt:
    d = yaml.safe_load(text)
    return Prompt(id=d["id"], version=str(d["version"]), description=d.get("description", ""), system=d["system"],
                  user=d["user"], temperature=float(d.get("temperature", 0.0)),
                  max_tokens=int(d.get("max_tokens", 2048)))


@lru_cache(maxsize=1)
def default_registry() -> PromptRegistry:
    folder = resources.files(__package__).joinpath("prompts")
    return PromptRegistry([_load(f.read_text(encoding="utf-8")) for f in sorted(folder.iterdir(), key=str)
                           if f.name.endswith(".yaml")])
