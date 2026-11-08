"""Phase 3: the briefing agent. Plan -> research (MCP evidence tools) -> write -> verify claim by claim -> gate.

Nodes are plain functions (``nodes``) so they are tested without a graph runtime. ``runner.run_briefing`` executes
them in order; ``graph.build_graph`` wires the same functions into LangGraph with a checkpointer and an
``interrupt()`` for human review (ADR 0001).
"""
