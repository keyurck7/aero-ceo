# Runbook: AERO-CEO v2 on SRH Data Lab

Data Lab (`compute.data-lab.site`) runs JupyterHub servers that are **billed per hour while running**.
Credits are the scarce resource, so: pick the cheapest node that fits the phase, and stop the server
as soon as a session ends.

## Which node, which image

Image: **PyTorch (LLM Focus)** (PyTorch, Hugging Face, vLLM, MCP and LangChain preinstalled).

| Work | Node | Cost | Why |
|---|---|---|---|
| Phases 1–2: import, BGE embeddings, reranker, retrieval evals, MCP servers | **London** or **Rome** (A6000 48 GB) | 10 credits/h | BGE-large plus a reranker need about 3 GB of VRAM; a 186-document corpus embeds in seconds |
| Phase 3 development: agent with **Qwen3.6-27B** | **Heidelberg** (Pro 6000 96 GB) | 12 credits/h | 27B BF16 weights are about 54 GB, leaving about 30 GB for KV cache (32K context) plus BGE |
| Phase 3 eval runs and demos (long context, throughput) | **Madrid** (H200 141 GB) | 15 credits/h | About 75 GB of headroom for KV cache; fastest memory |
| — | Frankfurt (2× A6000) | 10 credits/h | Avoid: the 2026-11-07 snapshot showed its GPU memory at about 95% used by other tenants |

At 170 credits, that is about 17 h on an A6000, 14 h on the Pro 6000 or 11 h on the H200.

## One-time setup (any node)

```bash
# The top of ~/vault is read-only; ~/vault/<project>/ may be writable and persists. ~ is lost on stop.
cd ~ && git clone -b v2 https://github.com/keyurck7/aero-ceo.git aero-ceo-v2 && cd aero-ceo-v2
python -m venv --system-site-packages .venv && source .venv/bin/activate   # reuse the image's torch/vLLM
pip install -e ".[mcp,extract,dev]" sentence-transformers
echo 'export HF_HOME=~/hf-cache' >> ~/.bashrc && export HF_HOME=~/hf-cache  # keep model downloads across spawns
df -h ~                                                                   # check the home quota before pulling a 54 GB model
python -m pytest -q   # NOT bare `pytest`: that is the image's copy and sees the image's old mcp 1.25
```

`.env` for this box (SQLite plus in-process models; Docker is normally unavailable inside JupyterHub):

```bash
AERO_ENV=local
AERO_DATABASE_URL=sqlite:///data/aero_v2.sqlite
AERO_EMBEDDING_BACKEND=sentence-transformers
AERO_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
AERO_EMBEDDING_DIM=1024
AERO_RERANKER_BACKEND=none          # measured 2026-11-08: the cross-encoder lowered nDCG@10 0.51 -> 0.38 on this corpus
AERO_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

## Phase 1 session: import the v1 corpus and measure retrieval (≈ 30 min on an A6000)

```bash
mkdir -p data && cp /path/to/aero_ceo.sqlite data/aero_ceo.sqlite
aero db migrate && aero seed sources
aero ingest v1 data/aero_ceo.sqlite          # 186 inserted, 180 templates stripped, 3 duplicates collapsed
aero search "How should Airbus respond if drone demand grows faster than fighter procurement?"

# Three-way ablation on the draft qrels (15 queries x 186 documents, labelled exhaustively)
AERO_RERANKER_BACKEND=none aero eval retrieval --qrels evals/datasets/retrieval_qrels.jsonl \
  --out evals/reports/retrieval_bge_norerank.json
aero eval retrieval --qrels evals/datasets/retrieval_qrels.jsonl --out evals/reports/retrieval_bge_rerank.json
```

Commit the two JSON summaries or paste them into `docs/v2-progress.md`. The hashing-embedder floor
measured in the sandbox is nDCG@10 0.42 / capped Recall@10 0.49 / MRR 0.77.

## After pulling phase 2 (signals, strategy, partners)

```bash
cd ~/aero-ceo-v2 && source .venv/bin/activate && export HF_HOME=~/hf-cache && git pull -q
aero db migrate                      # applies 0002
aero signals extract                 # expect ~191 signals
aero seed recommendations            # 6 drafts + candidate evidence (BGE)
aero rec list && aero rec show 1
```

## Phase 3 session: serve Qwen with vLLM

The image ships vLLM 0.13.0 (checked 2026-11-08), which is older than the documented minimum for Qwen3.6 (0.19).
Install a newer vLLM in the venv before phase 3, or use a Qwen3 model that 0.13 supports:

```bash
python -c "import vllm; print(vllm.__version__)"
```

Start the server in a terminal, or in the background with `nohup … &`:

```bash
vllm serve Qwen/Qwen3.6-27B \
  --served-model-name qwen3.6-27b \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --port 8000
```

- `--gpu-memory-utilization 0.80` leaves room for BGE and the reranker on the same GPU. On the H200,
  raise `--max-model-len` (e.g. 65536).
- Fallback if VRAM is tight: `Qwen/Qwen3.6-27B-FP8` (about half the weight memory).
- Check the model card for the current parser names before the first run; they change between vLLM
  releases.

Smoke test: `curl localhost:8000/v1/models`. The orchestrator (phase 3) reads `AERO_LLM_BASE_URL=http://localhost:8000/v1`.

## After pulling the no-heuristics change (ADR 0008) and phase 3

No re-import is needed: migration 0003 converts the database in place. It drops the rule-based signals, which
are derived data, and adds Admiralty source grades.

```bash
cd ~/aero-ceo && git pull && source .venv/bin/activate
pip install -e ".[mcp,extract,agent]"
# If your .env still has AERO_RETRIEVAL_CREDIBILITY_WEIGHT / _RECENCY_WEIGHT / _MIN_CREDIBILITY /
# AERO_RECENCY_HALF_LIFE_DAYS, delete those lines: they were removed and now fail with an explanation.
aero db migrate                 # applies 0003
aero sources grade              # Admiralty categories from seeds/sources.yaml -> grades A-F
aero search "FCAS workshare dispute"   # hits now show the source grade; grade E excluded by default
```

Then serve Qwen (section above) in a second terminal and point AERO-CEO at it:

```bash
export AERO_LLM_BASE_URL=http://localhost:8000/v1
export AERO_LLM_MODEL=qwen3.6-27b          # = --served-model-name
aero llm ping                              # must print structured_output {"ok": true, ...}

# Record every model call so the results are reproducible and reviewable (commit evals/cassettes later)
export AERO_LLM_CASSETTE_MODE=record_missing

aero signals extract --rejections data/signal_rejections.jsonl   # ~186 docs x (3 samples + verifications)
aero signals corroborate                                          # Admiralty information credibility 1/5/6
aero rec verify 1                                                 # candidate evidence -> supports/contradicts
aero partners rank "mission systems for uncrewed aircraft" --country Spain
aero brief "What does the FCAS dispute mean for Airbus, and what should we do next?"
```

Send back:
- the JSON printed by `aero signals extract`;
- `data/signal_rejections.jsonl`;
- the output of `aero brief`.

Labelling (the release gates in docs/plan-no-heuristics.md §7 need human labels):

```bash
aero eval sample-signals --n 60 --out evals/datasets/signal_gold_todo.jsonl
# label each row's "signals" (type, topic, direction, event_status, materiality), set "reviewed_by"
aero eval signals --gold evals/datasets/signal_gold.jsonl     # LLM vs gold, and vs the keyword baseline
aero eval judge                                               # verifier vs human labels (Cohen's kappa)
aero eval partners                                            # ranking vs analysts (Kendall tau-b)
```

## Credit discipline
- Stop the server from the Hub control panel when you finish. An idle kernel still bills.
- Keep `HF_HOME` and `data/` in your home directory, so a new spawn does not re-download or re-import.
- Prefer one long, planned session over many short ones (each spawn pays the model-load time).
