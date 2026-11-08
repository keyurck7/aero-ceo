# Plan: remove every heuristic weight from AERO-CEO v2

Status: requested by the project owner on 2026-11-08 ("I don't want to keep any heuristic weights"); being implemented on branch `v2`.
Decision record: [ADR 0008](adr/0008-no-heuristic-weights.md).

## 1. The rule

An executive briefing may only show four kinds of numbers or judgements:

| Kind | Example | What makes it legitimate |
|---|---|---|
| **Fact** | "3 independent publishers", "published 2026-10-14", "won 3 of 3 comparisons" | Counted or read, reproducible from the database |
| **Graded by a published standard** | Source reliability **B**, information credibility **1** (NATO Admiralty code) | The grade has a written definition, and the basis for each grade is recorded |
| **Model judgement, validated** | Signal direction "negative for Airbus", materiality "high", claim "supported" | JSON-schema output, an exact quote as grounding, an independent verification pass, and agreement with human labels measured (Cohen's κ) before it is trusted |
| **Calibrated probability** | "likely (55–80%)" | Fitted on labelled outcomes (isotonic regression) with the calibration error (ECE) reported; worded on the ICD 203 scale |

So the system contains **no hand-picked weight** that turns several factors into one score (no `0.6 × a + 0.2 × b`),
and **no invented float** dressed up as a measurement (no `credibility_prior = 0.84`, no `impact = 0.62`).

Engineering hyperparameters stay, because they control cost or search depth rather than encode a judgement.
Each one is listed in §4 with how it is validated.

## 2. Inventory: every heuristic and its replacement

Found in `src/aero` (v2) and in the local LangGraph agent (`v1-local-snapshot`, `app/agent_graph/`).

| # | Where | Heuristic today | Replacement | Kind |
|---|---|---|---|---|
| H1 | `scoring.evidence_credibility` | publisher prior × content factor (headline 0.8, page 0.95) × injection penalty 0.5 | **Deleted.** Source reliability grade (H2). `content_level` shown as a fact. Documents with prompt-injection flags are **quarantined**: excluded from retrieval unless explicitly requested, never down-weighted | Policy |
| H2 | `seeds/sources.yaml` `credibility_prior`, `publisher_priors`, `publisher_overrides`, `default_news_prior: 0.65` | Invented floats per publisher | **Admiralty source reliability A–F.** The analyst assigns each publisher a *category* (official primary source, established newsroom, specialist press, commentary, aggregator/UGC, off-topic/spam). A published table maps category → grade. Unreviewed publishers are **F (cannot be judged)** | Standard |
| H3 | `retrieval.blend_final`, `AERO_RETRIEVAL_CREDIBILITY_WEIGHT`, `…RECENCY_WEIGHT`, `recency_half_life_days` | Weighted blend of relevance, credibility, recency | **Deleted.** Rank = RRF fusion, or the cross-encoder order when the reranker is enabled *and* the retrieval eval shows it helps. Recency is a filter (`published_after`) and a displayed date, never a weight | Removed |
| H4 | `AERO_RETRIEVAL_MIN_CREDIBILITY = 0.15` | Threshold on an invented float | Policy: reliability **E** sources are excluded by default (`AERO_RETRIEVAL_EXCLUDE_RELIABILITY=E`); callers can ask for them explicitly | Policy |
| H5 | `RuleSignalExtractor` confidence `0.35 + 0.15·k (+0.15) × 0.85` | Formula | **Deleted.** LLM extraction + verification (H7). Self-consistency is reported as a fact, "found in 3 of 3 samples". A probability appears only after calibration (§5) | Fact |
| H6 | `taxonomy.topic_impact_weights`, `DEFAULT_IMPACT = 0.62`, `avg_impact`, `ORDER BY impact` | Invented per-topic priors | **Materiality** ordinal `high / medium / low` judged per signal against a **written rubric** (`seeds/materiality_rubric.yaml`), with a one-sentence rationale. Validated by κ against analyst labels. Ordering: materiality, then date | Validated model |
| H7 | Keyword rules for signals (**limit 1**: cannot read direction) | Keyword match | **LLM structured extraction**: `signal_type`, `direction_for_airbus` (positive/negative/mixed/neutral), `event_status` (announced … cancelled/failed), topic and business unit from the taxonomy enum, and an exact quote. Then: (a) deterministic grounding (the quote must be a substring of the chunk), (b) a coherence rule (an *opportunity* cannot be *negative* for Airbus; a *risk* cannot be *positive*), (c) an independent entailment check of the claim against the passage, (d) self-consistency over N samples. "What the failed fighter deal means" becomes an FCAS **risk**, direction negative, status failed | Validated model |
| H8 | `PartnerRegistry.match` = `0.6·capability + 0.2·country + 0.2·mentions` (**limit 2**) | Formula | (a) A required country is a **filter**, not a weight. (b) An LLM **rubric assessment** per partner (capability fit strong/partial/none, citing registry lines and evidence chunks); "none" is excluded. (c) **Pairwise comparisons** of the remaining partners, each pair judged in **both orders**; order-inconsistent verdicts become ties. (d) A **Bradley–Terry** model fitted to the outcomes gives the ranking. The output is the rank, the win/loss/tie record and the cited rationale, all facts. Validated by Kendall τ against analyst rankings | Fitted model |
| H9 | `taxonomy.infer_topics` keyword tagging at ingest | Keyword match | LLM multi-label topic classification over the taxonomy enum, done inside the extraction pass and validated per topic (P/R/F1). Keyword tags stay only as a provisional label (`topics_source = keyword`) until the LLM pass has run | Validated model |
| H10 | Approval gate counts *candidate* links | Unverified links count | Only links verified as `supports` (entailment judge or human), from sources not graded E, count towards "≥ 2 independent publishers" | Policy |
| H11 | Local agent `enterprise_guardrails`: claim "supported" if word overlap ≥ 0.08 / 0.14 | Thresholds | Entailment judgement per (claim, cited chunk): `supported / contradicted / not_enough_info`. Deterministic check that every number and date in a claim appears in a cited chunk. Unsupported claims are repaired or removed and **block** publication | Validated model + rule |
| H12 | Local agent `validators`: confidence = evidence/8 + signals/8 + … | Formula | No score until calibrated. The briefing header reports facts: claims verified x/y, independent publishers, contradictions found, Admiralty grades of the cited sources | Fact |
| H13 | Local agent `intent_classifier`: cosine thresholds against prototypes | Thresholds | LLM structured classification (`in_scope`, intent enum, reason), evaluated on the golden and out-of-scope sets (accuracy, false-accept rate on out-of-scope) | Validated model |
| H14 | Min-max normalisation of rerank scores | Existed only to blend | **Deleted** with the blend | Removed |

Also fixed while inventorying: the publisher **"Airbus"** (Google News items from Airbus) and **"Airbus Official
Web"** had different `publisher_group`s, so the independence check could count Airbus twice. Both are now group
`airbus`.

## 3. Standards used

**Source reliability (NATO Admiralty code, AJP-2.1 / STANAG 2511)**, assessed separately from the information:

| Grade | Name | AERO-CEO category that maps to it |
|---|---|---|
| A | Completely reliable | `official_primary`: an organisation's own channel speaking about its own decisions (Airbus, ministries, NATO, ESA, competitors about themselves). Authentic; says nothing about whether forward-looking claims come true |
| B | Usually reliable | `established_newsroom`: news organisation or specialist press with named editors and a corrections policy |
| C | Fairly reliable | `specialist_or_commentary`: smaller trade titles, analyst blogs and commentary outlets with named authors |
| D | Not usually reliable | `aggregator_or_ugc`: press-release redistribution, finance community platforms, content aggregators |
| E | Unreliable | `off_topic_or_spam`: off-topic or spam content in this corpus, or a history of invalid information |
| F | Cannot be judged | `unreviewed`: default for any new publisher until an analyst assigns a category |

**Information credibility (Admiralty 1–6)**, per signal, from corroboration only:

| Observed fact | Grade |
|---|---|
| ≥ 1 *other* publisher group (not grade E) has a passage judged to **support** the claim, and none contradicts it | **1** Confirmed by other sources |
| ≥ 1 other publisher group **contradicts** it, none supports it | **5** Improbable |
| Supported *and* contradicted by others | **6** Cannot be judged, flagged `contested`, both sides shown |
| No other publisher addresses it | **6** Cannot be judged (single source) |

Grades 2–4 need an analyst's judgement of logic and consistency, so the system never assigns them automatically.
An analyst can.

**Estimative language (ICD 203).** Only a *calibrated* probability is turned into words: almost no chance
(1–5%), very unlikely (5–20%), unlikely (20–45%), roughly even chance (45–55%), likely (55–80%), very likely
(80–95%), almost certain (95–99%). Without calibration the briefing states facts and does not use these words.

**LLM-as-judge validation.**
- Every judge prompt is versioned.
- Each judge has a gold set labelled by a human.
- Agreement is reported as Cohen's κ, with κ ≥ 0.6 ("substantial", Landis & Koch) required before the judge gates anything.
- Pairwise judges run in both orders to cancel position bias.
- The judge prompt differs from the generator prompt. A second model family will be added as judge when one is available on Data Lab.

## 4. What remains a parameter, and why that is allowed

| Parameter | Value | Role | How it is validated |
|---|---|---|---|
| RRF `k` | 60 | Rank fusion constant (Cormack et al., 2009) | Retrieval eval; the Data Lab ablation |
| Candidate depth | 50 | Search depth (cost) | Recall@50 on qrels |
| Chunk size / overlap | 300 / 45 words | Unit of evidence | Retrieval eval |
| SimHash Hamming bound | 7 | Near-duplicate detection | Measured on the corpus; refit on labelled duplicate pairs (P/R) |
| Self-consistency samples `N` | 3 | Cost vs stability | Agreement statistics in the extraction report |
| Corroboration depth | 5 passages | Search depth | Recall of confirming passages on the claim gold set |
| Majority rule (> N/2) | — | Standard self-consistency decision rule | Extraction P/R vs gold |
| Approval: ≥ 2 independent publishers | — | Editorial two-source rule (policy) | Reviewed by the owner |

## 5. Calibration: when a probability may be shown

1. Each reviewed briefing claim and each reviewed signal yields a labelled pair (system judgement, human verdict).
2. Once at least 200 labels exist per judge, fit isotonic regression from observed features (self-consistency
   count, verification verdict, corroboration grade) to correctness. Report ECE and a reliability diagram on a
   held-out split.
3. Ship the probability only if ECE ≤ 0.05 on the held-out split. Otherwise keep showing facts.

The code for κ, F1, Kendall τ, ECE and isotonic fitting is in `aero.evals.metrics` and is tested now. Fitting
waits for labels.

## 6. Labelling plan (owner review needed)

| Set | Size | Who | Status |
|---|---|---|---|
| Source categories | 78 publishers | Claude drafted, owner reviews | Drafted in `seeds/sources.yaml` |
| Signal gold | 60 documents, stratified by content level and topic | Owner (Claude can pre-label; only human-reviewed rows count) | Sampler: `aero eval sample signals` |
| Materiality | Same 60 documents | Owner | Rubric drafted |
| Partner rankings | 10 needs × top-5 order | Owner | Template in `evals/datasets/partner_rankings.jsonl` |
| Claim verification | 100 (claim, passage) pairs from system output | Owner | After first Qwen run |
| Out-of-scope | 5 → 30 questions | Owner | Seed exists |

## 7. Release gates (an executive-level agent ships only when all hold)

| Gate | Target |
|---|---|
| Quote grounding of stored signals | 100% (deterministic; enforced in code) |
| Signal extraction on gold (type + topic + chunk), F1 | ≥ 0.70, and better than the rule baseline |
| Direction accuracy on gold | ≥ 0.85 |
| Judge–human agreement (verification, materiality) | κ ≥ 0.6 |
| Partner ranking vs analysts | mean Kendall τ ≥ 0.5 |
| Unsupported claims reaching a briefing (golden set) | 0 unlabelled; ≤ 2% before repair |
| Out-of-scope false accepts | 0 on the out-of-scope set |
| Probabilities shown | only with ECE ≤ 0.05 |

The thresholds are release gates set by the owner, the same way an SLO is. They are not inside any score.

## 8. Order of work

1. Admiralty grading. Delete the blend and its settings. Migration 0003 (forward-only, no re-import needed).
2. `aero.llm`: OpenAI-compatible client for vLLM, JSON-schema structured outputs, versioned prompt registry,
   record/replay cassettes, and a fake LLM for tests.
3. `aero.intelligence.verification`: entailment judge (signals, corroboration, recommendation evidence, briefing claims).
4. LLM signal extractor (direction, status, materiality, grounding, verification, self-consistency) and
   corroboration into Admiralty credibility. The rule extractor moves to `aero.evals.baselines`.
5. Partner ranking: rubric assessment, pairwise judgements in both orders, Bradley–Terry.
6. Eval harness: metrics, signal/partner/judge evaluations, samplers for labelling.
7. Phase 3 orchestrator on top of the same LLM layer and verification module.
8. Data Lab: serve Qwen, run the jobs, record cassettes, label, then check the release gates.

## Sources
- [Admiralty code (Wikipedia)](https://en.wikipedia.org/wiki/Admiralty_code)
- [The Admiralty Code / NATO 6x6 System](https://pangearesearch.substack.com/p/the-admiralty-code-nato-6x6-system)
- [SANS: Enhance your CTI with the Admiralty System](https://sans.org/blog/enhance-your-cyber-threat-intelligence-with-the-admiralty-system)
- [ICD 203 seven-point probability scale](https://wphillipmoore.github.io/the-infrastructure-mindset/research/R0052-methodology-claims/2026-03-31-02/C009-icd203-probability-scale/assessment)
- [NATO STO SAS-114: assessment and communication of uncertainty](https://www.sto.nato.int/publications/STO%20Meeting%20Proceedings/STO-MP-SAS-114/MP-SAS-114-ANN-B.pdf)
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models* (2022)
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023): position bias and swapping
- Bradley & Terry (1952); Hunter, *MM algorithms for generalized Bradley–Terry models* (2004)
- Cormack, Clarke & Büttcher, *Reciprocal Rank Fusion* (2009)
- Landis & Koch (1977) for κ bands; Zadrozny & Elkan (2002) for isotonic calibration
