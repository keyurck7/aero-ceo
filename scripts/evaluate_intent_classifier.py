"""
Evaluate AERO-CEO intent and scope classifier on a small labeled test set.

This is not a full production benchmark, but it gives a repeatable check
for domain scope behavior and core intent routing.
"""

from __future__ import annotations

from app.agent_graph.intent_classifier import classify_goal


TEST_CASES = [
    {
        "query": "NLP exam",
        "expected_scope": "out_of_scope",
        "expected_intent": "out_of_scope",
    },
    {
        "query": "Write a poem about my birthday",
        "expected_scope": "out_of_scope",
        "expected_intent": "out_of_scope",
    },
    {
        "query": "What should Airbus do next for FCAS and uncrewed systems?",
        "expected_scope": "in_scope",
        "expected_intent": None,
    },
    {
        "query": "Which European organization should Airbus collaborate with for sixth-generation fighter systems?",
        "expected_scope": "in_scope",
        "expected_intent": "partnership_strategy",
    },
    {
        "query": "What are the biggest risks for Airbus Defence and Space right now?",
        "expected_scope": "in_scope",
        "expected_intent": "risk_analysis",
    },
    {
        "query": "What opportunities should Airbus prioritize in military space?",
        "expected_scope": "in_scope",
        "expected_intent": "opportunity_analysis",
    },
    {
        "query": "What if FCAS is delayed by five years?",
        "expected_scope": "in_scope",
        "expected_intent": "scenario_analysis",
    },
    {
        "query": "Show evidence for Airbus military space recommendations",
        "expected_scope": "in_scope",
        "expected_intent": "evidence_request",
    },
]


def main() -> None:
    scope_pass = 0
    intent_pass = 0
    intent_total = 0

    print("=" * 90)
    print("AERO-CEO INTENT CLASSIFIER EVALUATION")
    print("=" * 90)

    for i, case in enumerate(TEST_CASES, start=1):
        result = classify_goal(case["query"])

        scope_ok = result.get("scope_status") == case["expected_scope"]
        scope_pass += int(scope_ok)

        expected_intent = case["expected_intent"]
        intent_ok = True
        if expected_intent is not None:
            intent_total += 1
            intent_ok = result.get("intent") == expected_intent
            intent_pass += int(intent_ok)

        print(f"\nTest {i}: {case['query']}")
        print(f"  Expected scope: {case['expected_scope']} | Got: {result.get('scope_status')} | OK={scope_ok}")
        print(f"  Expected intent: {expected_intent} | Got: {result.get('intent')} | OK={intent_ok}")
        print(f"  Topic: {result.get('topic')} | Domain relevance: {result.get('domain_relevance')}")
        print(f"  Reason: {result.get('rejection_reason')}")

    print("\n" + "=" * 90)
    print(f"Scope accuracy: {scope_pass}/{len(TEST_CASES)}")
    print(f"Intent accuracy on checked cases: {intent_pass}/{intent_total}")
    print("=" * 90)


if __name__ == "__main__":
    main()
