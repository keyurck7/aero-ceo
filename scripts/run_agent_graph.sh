#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

python -m app.agent_graph.run_agent_goal "$@"
