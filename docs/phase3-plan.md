# Phase 3 plan: port the local LangGraph agent onto MCP tools and Qwen

Source: branch `v1-local-snapshot`, `app/agent_graph/` (about 3,300 lines, written on Data Lab, never pushed before).
Target: `src/aero/orchestrator/`, calling MCP servers and a vLLM-served Qwen model.

## What the local graph already does (keep the design)

`classify_goal → (out_of_scope_response | build_plan) → retrieve_evidence → analyze_context → decide_strategy →
validate_output → write_briefing → verify_briefing → save_trace`, with:
- an embedding-prototype scope guard (it refused "NLP exam" 5 times in `agent_traces`);
- per-node trace entries and a SQLite trace and approval store;
- a deterministic fallback briefing if the LLM fails;
- a claim–citation map, contradiction pairs, an optional LLM judge, approval-reason rules and JSONL observability.

## Node-by-node port

| Local node / module | v2 replacement | Change |
|---|---|---|
| `intent_classifier.classify_goal` (loads its own BGE) | `nodes/understand.py` + `nodes/scope_guard.py` | Keep the prototype idea; reuse the shared embedder; Qwen structured output with keyword fallback; prototypes move to `seeds/taxonomy.yaml` |
| `build_plan` (static list, display only) | `nodes/plan.py` | Plan **drives** execution: sub-questions, tool choices and a stop rule per sub-question |
| `tools.semantic_retrieval` (`goal + "Airbus" + topic + area` concatenated) | `search_evidence` on evidence-mcp | One query per sub-question; no keyword stuffing; filters instead of appended words |
| `tools.lookup_signals` / `lookup_recommendations` / `analyze_partners` | signals-mcp / strategy-mcp / partners-mcp | Built next from `seeds/` (phase 2b) |
| `retrieve → analyze` (single pass) | research loop + `coverage_check` | Up to 2 rounds; at least 2 independent publishers per sub-question; uses `independent_publishers` from evidence-mcp |
| `validators.validate_agent_output` (confidence = evidence/8 + signals/8 + …) | `nodes/gate.py` + calibrated confidence | Confidence from **claim support**, publisher independence, credibility and contradictions (architecture §10.2) |
| `executive_writer` (Qwen2.5-3B, transformers in-process) | `nodes/write.py` → vLLM | Qwen3.6-27B; `[C#]` chunk citations; no tools bound; token budget; evidence fenced as untrusted |
| `post_generation_verifier` + `enterprise_guardrails._map_claims_to_citations` (8–14% word overlap = "supported") | verify-mcp `verify_claims` | Entailment judged by Qwen (and/or an NLI model) per claim × cited chunk; numbers and dates must appear in a cited chunk |
| guardrail FAILED → `APPROVED_WITH_WARNING` | `repair` edge → re-research once → drop unsupported claims | **Verification can block**: unsupported claims never reach the CEO unlabelled |
| "Human review note" appended to text | LangGraph `interrupt()` + SQLite/Postgres checkpointer | Real pause and resume; decisions stored in `approvals`; Mission Control becomes the reviewer UI |
| `_append_trace` + JSONL logs | OpenTelemetry spans (later Langfuse) + `agent_runs` table | Same information, standard format |
| `graph.py.broken_backup` | — | Not ported |

## Model serving on Data Lab
- The image ships **vLLM 0.13.0**, older than the documented minimum for Qwen3.6 (0.19).
- Plan A: `pip install -U vllm` inside the venv, on a node with persistent space for the wheel and model cache.
- Plan B (if A fails): a Qwen model that vLLM 0.13 already serves, e.g. `Qwen/Qwen3-32B-FP8` on the A6000 or
  `Qwen/Qwen3-30B-A3B-Instruct-2507` on the Pro 6000, behind the same `AERO_LLM_BASE_URL`.
- The orchestrator only sees an OpenAI-compatible endpoint, so the A/B choice changes no code.

## Order of work
1. signals-mcp, strategy-mcp (+ the 7 drill-down prompts) and partners-mcp from `seeds/` (no GPU needed).
2. Orchestrator skeleton with a fake LLM in tests; port the nodes in the table above.
3. Data Lab session: serve Qwen; run the 10 golden questions and the 5 out-of-scope questions; record
   tool-selection, trajectory and faithfulness metrics.
4. verify-mcp + repair loop + `interrupt()` approvals.
