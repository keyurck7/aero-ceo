# ADR 0001: MCP servers for capabilities, LangGraph for control flow

**Status:** accepted · 2026-11-07

## Context
v1 is a fixed function chain: route → retrieve → template → optional LLM rewrite. It cannot plan,
re-research or verify, and none of its capabilities can be reused by other agents or hosts.

## Decision
- Every capability (evidence, signals, strategy, partners, verification, reports, ingest-ops) is a
  **stateless MCP server** (official SDK v2, protocol 2026-07-28, Streamable HTTP). Each has its own
  least-privilege DB role and scales independently.
- The agent is a **LangGraph state machine** that calls those servers through
  `langchain-mcp-adapters`. Each node binds only the small tool set it needs.
- Conversation state lives in the LangGraph Postgres checkpointer, never in MCP connections.

## Consequences
- Any MCP host (Claude Desktop, IDE agents, another team's agent) can use the same tools with the same
  audit and auth.
- Two moving parts instead of one; contract tests per server keep the interface stable.
- First server delivered: `evidence-mcp` (`src/aero/mcp_servers/evidence.py`).
