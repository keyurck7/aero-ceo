#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

python -m streamlit run app/dashboard/streamlit_app.py \
  --server.port "${STREAMLIT_PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
