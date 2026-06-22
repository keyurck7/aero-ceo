#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

python - <<'PY'
from app.agents.local_llm import get_local_llm

llm = get_local_llm()

messages = [
    {
        "role": "system",
        "content": "You are AERO-CEO, a concise strategic intelligence advisor for Airbus SE."
    },
    {
        "role": "user",
        "content": "In 5 bullet points, explain why evidence grounding matters in a CEO strategy agent."
    }
]

answer = llm.generate(messages, max_new_tokens=300)
print("\n=== LLM TEST ANSWER ===\n")
print(answer)
PY
