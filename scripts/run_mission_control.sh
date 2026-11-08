#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

python -m streamlit run app/dashboard/ceo_mission_control.py \
  --server.port "${STREAMLIT_PORT:-8502}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
