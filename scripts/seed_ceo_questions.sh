#!/usr/bin/env bash
set -Eeuo pipefail

echo "=== Seeding CEO Q&A demo questions ==="

# Keep rebuild fast. Live Streamlit can still use LLM after rebuild.
export LLM_GENERATION_ENABLED=false

bash scripts/ask_ceo.sh "Should Airbus collaborate with another European organization for sixth-generation fighter systems?"
bash scripts/ask_ceo.sh "What are the biggest risks if FCAS is delayed?"
bash scripts/ask_ceo.sh "What should Airbus do if budget is limited but it still wants to compete in uncrewed combat aircraft?"
bash scripts/ask_ceo.sh "Explain the recommendation about military space and secure communications with evidence."
bash scripts/ask_ceo.sh "Which partner could help Airbus in Spain for SIRTAP and uncrewed systems?"
bash scripts/ask_ceo.sh "What is the lower-risk version of investing in future combat systems?"

echo "=== CEO Q&A demo questions seeded ==="
