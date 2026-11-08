#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CLEAN=false
SKIP_COLLECTION=false
SKIP_FAISS=false
SKIP_QNA=false
SKIP_EVAL=false

for arg in "$@"; do
  case "$arg" in
    --clean)
      CLEAN=true
      ;;
    --skip-collection)
      SKIP_COLLECTION=true
      ;;
    --skip-faiss)
      SKIP_FAISS=true
      ;;
    --skip-qna)
      SKIP_QNA=true
      ;;
    --skip-eval)
      SKIP_EVAL=true
      ;;
    --help|-h)
      echo "Usage: bash scripts/rebuild_demo.sh [--clean] [--skip-collection] [--skip-faiss] [--skip-qna] [--skip-eval]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Use --help for options."
      exit 1
      ;;
  esac
done

section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

run() {
  echo "+ $*"
  "$@"
}

check_file() {
  if [ ! -f "$1" ]; then
    echo "Missing required file: $1"
    exit 1
  fi
}

section "AERO-CEO rebuild started"

echo "Project root: $(pwd)"
echo "Python: $($PYTHON_BIN --version)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

section "Preflight checks"

check_file "app/db/local_db.py"
check_file "app/ingestion/collectors/airbus_live_collector.py"
check_file "app/ingestion/collectors/rss_intelligence_collector.py"
check_file "app/retrieval/build_faiss_index.py"
check_file "app/intelligence/signal_extractor.py"
check_file "app/agents/ceo_recommendation_engine.py"
check_file "app/dashboard/streamlit_app.py"

mkdir -p data/raw data/processed data/cache data/demo_dataset reports logs scripts

if [ ! -f ".env" ]; then
  section "Creating minimal .env"
  if [ -f ".env.example" ]; then
    cp .env.example .env
  else
    cat > .env <<'ENV'
APP_ENV=local
LOCAL_DEV_MODE=true
DATABASE_MODE=sqlite
SQLITE_DB_PATH=data/aero_ceo.sqlite
FAISS_INDEX_PATH=data/cache/aero_ceo.faiss
FAISS_METADATA_PATH=data/cache/faiss_metadata.json
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
LLM_GENERATION_ENABLED=false
LLM_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
LLM_MAX_NEW_TOKENS=700
LLM_TEMPERATURE=0.25
LLM_TOP_P=0.90
CUDA_VISIBLE_DEVICES=1
ENV
  fi
fi

if [ "$CLEAN" = true ]; then
  section "Clean rebuild requested"
  rm -f data/aero_ceo.sqlite
  rm -f data/cache/aero_ceo.faiss
  rm -f data/cache/faiss_metadata.json
  rm -f reports/evaluation_report.json
  echo "Removed generated SQLite, FAISS, metadata, and evaluation report files."
fi

section "Step 1/8: Initialize local database"
run "$PYTHON_BIN" app/db/local_db.py

if [ "$SKIP_COLLECTION" = false ]; then
  section "Step 2/8: Collect Airbus official/news intelligence"
  run "$PYTHON_BIN" -m app.ingestion.collectors.airbus_live_collector \
    --official-limit 40 \
    --news-limit 180 \
    --per-query 25

  section "Step 3/8: Collect supplemental RSS intelligence"
  run "$PYTHON_BIN" -m app.ingestion.collectors.rss_intelligence_collector \
    --per-query 20 \
    --target 180
else
  section "Step 2/8 and 3/8: Collection skipped"
fi

section "Corpus summary after collection"
"$PYTHON_BIN" - <<'PY'
from app.db.local_db import get_connection

conn = get_connection()
cur = conn.cursor()

def count(table):
    try:
        return cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except Exception:
        return 0

print("Documents:", count("documents"))
print("Sources:", count("sources"))

try:
    print("\nTop topics:")
    for row in cur.execute("""
        SELECT topic, COUNT(*) AS n
        FROM documents
        GROUP BY topic
        ORDER BY n DESC
        LIMIT 10
    """):
        print(row["topic"], row["n"])
except Exception as exc:
    print("Topic summary unavailable:", exc)

conn.close()
PY

if [ "$SKIP_FAISS" = false ]; then
  section "Step 4/8: Build FAISS semantic index"
  run "$PYTHON_BIN" -m app.retrieval.build_faiss_index
else
  section "Step 4/8: FAISS rebuild skipped"
fi

section "Step 5/8: Extract strategic signals"
run "$PYTHON_BIN" -m app.intelligence.signal_extractor --reset

section "Step 6/8: Generate CEO recommendations"
run "$PYTHON_BIN" -m app.agents.ceo_recommendation_engine

if [ "$SKIP_QNA" = false ]; then
  section "Step 7/8: Seed CEO Q&A demo questions"
  if [ -f "scripts/ask_ceo.sh" ]; then
    run bash scripts/seed_ceo_questions.sh
  else
    echo "scripts/ask_ceo.sh not found. Skipping CEO Q&A seeding."
  fi
else
  section "Step 7/8: CEO Q&A seeding skipped"
fi

if [ "$SKIP_EVAL" = false ]; then
  section "Step 8/8: Run evaluation and guardrails"
  if [ -f "app/evaluation/evaluation_suite.py" ]; then
    run "$PYTHON_BIN" -m app.evaluation.evaluation_suite --save
  else
    echo "Evaluation suite not found. Skipping evaluation."
  fi
else
  section "Step 8/8: Evaluation skipped"
fi

section "Final system summary"
"$PYTHON_BIN" - <<'PY'
from app.db.local_db import get_connection

conn = get_connection()
cur = conn.cursor()

def count(table):
    try:
        return cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    except Exception:
        return 0

print("Documents:", count("documents"))
print("Sources:", count("sources"))
print("Chunks:", count("document_chunks"))
print("Signals:", count("signals"))
print("Recommendations:", count("recommendations"))
print("Recommendation evidence links:", count("recommendation_evidence"))
print("CEO Q&A sessions:", count("ceo_queries"))

try:
    print("\nTop recommendations:")
    for row in cur.execute("""
        SELECT id, priority, confidence_score, title
        FROM recommendations
        ORDER BY
            CASE priority
                WHEN 'High' THEN 1
                WHEN 'Medium' THEN 2
                ELSE 3
            END,
            confidence_score DESC
        LIMIT 10
    """):
        print(row["id"], "|", row["priority"], "|", row["confidence_score"], "|", row["title"])
except Exception as exc:
    print("Recommendation summary unavailable:", exc)

conn.close()
PY

section "AERO-CEO rebuild complete"
echo "Run dashboard:"
echo "  bash scripts/run_streamlit.sh"
echo
echo "Check dashboard server from a second terminal:"
echo "  curl -I http://127.0.0.1:8501"
echo
echo "Useful options:"
echo "  bash scripts/rebuild_demo.sh --clean"
echo "  bash scripts/rebuild_demo.sh --skip-collection"
echo "  bash scripts/rebuild_demo.sh --skip-qna"
