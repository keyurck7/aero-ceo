# Evaluation datasets

| File | What | Status |
|---|---|---|
| `retrieval_queries.jsonl` | 15 curated retrieval queries: v1 probes, v1 demo questions, and the two most-asked goals in v1 `agent_traces` (11× and 6×) | curated, source of truth |
| `retrieval_qrels.jsonl` | Graded relevance (2 = direct, 1 = partial) for those 15 queries over **all 186 documents** of the v1 import, keyed by canonical URL; 324 judgements | **draft**: labelled by Claude from titles; needs owner review |
| `golden_questions.jsonl` | CEO questions for agent evals (phase 3); key points still to write | curated, needs labels |
| `out_of_scope_seed.jsonl` | Questions the scope guard must refuse; "NLP exam" was entered 5× as an agent goal in v1 | seed |
| `*_seed.jsonl` (others) | Regenerated from v1 code by `scripts/v2/extract_v1_seeds.py` | generated, do not edit |

## Reviewing the draft qrels
Labels were judged exhaustively: every document title, against every query. Because no retriever chose
the candidates, the set is not biased towards any one retriever. Review checklist:
1. `aero eval pool --queries evals/datasets/retrieval_queries.jsonl --k 20` lists what the current
   retriever returns. Any highly ranked document that is missing from the qrels deserves a second look.
2. Change a grade in `retrieval_qrels.jsonl`, then set `"label_status": "reviewed"` on that query.
3. The qrels are tied to corpus `v1-import-2026-11-07` (Google-News canonical URLs). After re-collecting
   news, re-pool and re-label the new documents.
