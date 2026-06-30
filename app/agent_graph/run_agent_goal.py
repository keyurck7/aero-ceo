"""
CLI runner for deterministic AERO-CEO LangGraph workflow.
"""

from __future__ import annotations

import argparse
import json

from app.agent_graph.graph import run_agent_goal


def main():
    parser = argparse.ArgumentParser(description="Run AERO-CEO deterministic agent graph.")
    parser.add_argument("goal", nargs="*", help="CEO goal/question.")
    parser.add_argument("--json", action="store_true", help="Print full final state as JSON.")
    args = parser.parse_args()

    goal = " ".join(args.goal).strip()
    if not goal:
        goal = "If you were Airbus CEO today, what would you do next and why?"

    state = run_agent_goal(goal)

    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
        return

    print("\n" + "=" * 90)
    print("AERO-CEO DETERMINISTIC LANGGRAPH RUN")
    print("=" * 90)

    print(f"\nGoal: {state.get('goal')}")
    print(f"Run ID: {state.get('run_id')}")
    print(f"Intent: {state.get('intent')} | Confidence: {state.get('intent_confidence')}")
    print(f"Topic: {state.get('topic')} | Confidence: {state.get('topic_confidence')}")
    print(f"Business area: {state.get('business_area')}")
    print(f"Validation: {state.get('validation', {}).get('status')} | Confidence: {state.get('validation', {}).get('confidence')}")

    print("\n--- Agent Plan ---")
    for step in state.get("plan", []):
        print(f"{step.get('step')}. {step.get('name')} -> {step.get('purpose')}")

    print("\n--- Tool Trace ---")
    for t in state.get("tool_trace", []):
        print(f"[{t.get('step')}] {t.get('tool')} -> {t.get('status')}")
        print(f"  {t.get('observation')}")

    print("\n--- Final CEO Briefing ---")
    print(state.get("briefing", ""))


if __name__ == "__main__":
    main()
