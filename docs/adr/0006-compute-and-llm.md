# ADR 0006: Compute on SRH Data Lab; Qwen3.6-27B on vLLM, local only

**Status:** accepted · 2026-11-07

## Context
- Constraints from the project owner: academic project; **Qwen models only**; no hosted LLM APIs.
- Compute: SRH Data Lab JupyterHub with per-hour credit billing. Image "PyTorch (LLM Focus)" ships vLLM.
  Nodes: A6000 48 GB (10 credits/h), Pro 6000 96 GB (12/h), H200 141 GB (15/h).
- v1 used Qwen2.5-3B in-process via `transformers`. Its prompt truncation cut the question off
  (finding B2), and 3B is too small for reliable multi-step tool use.

## Decision
- **Writer/agent model:** `Qwen/Qwen3.6-27B` (dense, Apache 2.0, released 2026-04-22), BF16, served by
  vLLM's OpenAI-compatible server with tool calling and the reasoning parser enabled. One model serves
  every graph node in phase 3. The structured-output nodes use guided JSON decoding.
- **Alternative to ablate in phase 3:** `Qwen/Qwen3.6-35B-A3B` (MoE, about 3B active), which is faster
  per token. Pick by golden-set faithfulness and tool-selection scores, not by size.
- **Node policy:** A6000 for phases 1–2 (embeddings only); Pro 6000 96 GB for agent development;
  H200 for eval runs and demos. Details in `docs/runbooks/datalab.md`.
- **No LiteLLM gateway yet:** with a single local model, the orchestrator talks to vLLM's OpenAI-compatible
  endpoint directly through `AERO_LLM_BASE_URL`. Revisit if a second model or a budget limit is needed.
- **Storage on Data Lab:** the SQLite adapter (Docker is normally unavailable inside JupyterHub);
  Postgres + pgvector stays the shared/production target and is tested in CI.

## Consequences
The architecture document's "LiteLLM model gateway" (§9) is deferred, not dropped. The model choice
must be re-validated on the golden set once phase-3 evals exist.
