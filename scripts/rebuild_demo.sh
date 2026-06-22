#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

python app/db/local_db.py
python -m app.ingestion.collectors.airbus_live_collector --official-limit 40 --news-limit 180 --per-query 25
python -m app.ingestion.collectors.rss_intelligence_collector --per-query 20 --target 180
python -m app.retrieval.build_faiss_index
python -m app.intelligence.signal_extractor --reset
python -m app.agents.ceo_recommendation_engine
bash scripts/seed_ceo_questions.sh
python -m app.dashboard.generate_static_report
python -m app.dashboard.generate_ceo_qna_report
