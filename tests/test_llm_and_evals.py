"""The LLM layer (client, prompts, structured outputs, cassettes) and the judgement evaluations/metrics."""

from __future__ import annotations

import json

import pytest
import requests
from pydantic import BaseModel

from aero.config import load_settings
from aero.evals import judgements as jd
from aero.evals.baselines import RuleSignalExtractor, load_signal_rules, run_baseline
from aero.evals.metrics import (
    cohen_kappa,
    expected_calibration_error,
    fit_isotonic,
    icd203_term,
    kendall_tau_b,
    prf,
    ranking_tau,
)
from aero.intelligence.partners import PartnerRanker, PartnerRegistry
from aero.intelligence.signals import LLMSignalExtractor, load_materiality_rubric, run_extraction
from aero.intelligence.verification import Verifier
from aero.llm import LLMError, LLMRequest, LLMUnavailableError, OpenAICompatibleLLM, build_llm, generate
from aero.llm.cassette import CassetteLLM, request_key
from aero.llm.fake import FakeLLM
from aero.llm.prompts import Prompt, PromptRegistry, default_registry
from aero.llm.structured import StructuredOutputError
from tests.conftest import ROOT
from tests.fakes import scripted_llm

SEEDS = ROOT / "seeds"


class Answer(BaseModel):
    verdict: str
    n: int


PROMPT = Prompt(id="t", version="2", description="", system="Judge {{thing}}.", user="Input: {{x}}")


# ----------------------------------------------------------------- prompts
def test_prompt_render_is_strict():
    msgs = PROMPT.render({"thing": "claims", "x": "$5bn {not a var}"})
    assert msgs == [{"role": "system", "content": "Judge claims."},
                    {"role": "user", "content": "Input: $5bn {not a var}"}]
    with pytest.raises(ValueError, match="missing"):
        PROMPT.render({"thing": "claims"})
    with pytest.raises(ValueError, match="unknown"):
        PROMPT.render({"thing": "a", "x": "b", "y": "c"})
    assert PROMPT.ref == "t@2"


def test_packaged_prompts_are_versioned_and_complete():
    reg = default_registry()
    assert reg.ids() == ["brief_write", "partner_assess", "partner_compare", "scope_plan", "signal_extract",
                         "verify_claim"]
    assert reg.get("signal_extract").variables == {
        "materiality_rubric", "topics", "business_units", "publisher", "published_at", "content_level", "title",
        "chunks"}
    assert all(reg.get(i).version for i in reg.ids())
    with pytest.raises(KeyError, match="unknown prompt"):
        reg.get("nope")
    with pytest.raises(ValueError, match="duplicate"):
        PromptRegistry([PROMPT, PROMPT])


# ----------------------------------------------------------------- structured generation
def test_generate_validates_and_repairs_once():
    replies = iter(['```json\n{"verdict": "ok"}\n```', '<think>hmm</think>{"verdict": "ok", "n": 3}'])
    llm = FakeLLM(lambda req: next(replies))
    out, _ = generate(llm, PROMPT, Answer, {"thing": "a", "x": "b"})
    assert out == Answer(verdict="ok", n=3)
    first, repair = llm.calls
    assert first.schema == Answer.model_json_schema() and first.schema_name == "Answer"
    assert repair.messages[-1]["role"] == "user" and "did not validate" in repair.messages[-1]["content"]


def test_generate_fails_loudly_after_repair():
    llm = FakeLLM(lambda req: "not json")
    with pytest.raises(StructuredOutputError, match="t@2"):
        generate(llm, PROMPT, Answer, {"thing": "a", "x": "b"})


# ----------------------------------------------------------------- HTTP client
class FakeResponse:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.bodies = []

    def post(self, url, data, headers, timeout):
        self.bodies.append((url, json.loads(data), headers))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _req(**kw):
    base = dict(prompt_id="t", prompt_version="1", messages=[{"role": "user", "content": "hi"}], model="qwen",
                schema_name="Answer", schema=Answer.model_json_schema(), temperature=0.7, seed=2, top_p=0.8)
    return LLMRequest(**{**base, **kw})


def test_openai_client_sends_json_schema_and_retries(monkeypatch):
    monkeypatch.setattr("aero.llm.client.time.sleep", lambda s: None)
    ok = FakeResponse(200, {"model": "qwen", "choices": [{"message": {"content": '{"verdict":"x","n":1}'},
                                                           "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 3}})
    session = FakeSession([requests.ConnectionError("down"), FakeResponse(503, {"error": "busy"}), ok])
    client = OpenAICompatibleLLM("http://vllm:8000/v1/", "qwen", api_key="k", max_retries=2, session=session)
    c = client.complete(_req())
    assert c.text == '{"verdict":"x","n":1}' and c.usage == {"prompt_tokens": 5, "completion_tokens": 3}
    url, body, headers = session.bodies[-1]
    assert url == "http://vllm:8000/v1/chat/completions" and headers["Authorization"] == "Bearer k"
    assert body["response_format"]["json_schema"]["schema"] == Answer.model_json_schema()
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert (body["seed"], body["top_p"], body["temperature"]) == (2, 0.8, 0.7)


def test_openai_client_errors():
    bad = OpenAICompatibleLLM("http://x/v1", "q", max_retries=0, session=FakeSession([FakeResponse(400, {"e": 1})]))
    with pytest.raises(LLMError, match="HTTP 400"):
        bad.complete(_req())
    down = OpenAICompatibleLLM("http://x/v1", "q", max_retries=0, session=FakeSession([requests.Timeout("t")]))
    with pytest.raises(LLMUnavailableError):
        down.complete(_req())


def test_unconfigured_llm_refuses_with_guidance():
    llm = build_llm(load_settings({}, dotenv_path=None))
    with pytest.raises(LLMUnavailableError, match="AERO_LLM_BASE_URL"):
        llm.complete(_req())


# ----------------------------------------------------------------- cassettes
def test_cassette_record_then_replay(tmp_path):
    live = FakeLLM(lambda req: '{"verdict": "live", "n": 1}')
    rec = CassetteLLM(live, tmp_path, "record")
    assert rec.complete(_req()).text == '{"verdict": "live", "n": 1}'
    replay = CassetteLLM(FakeLLM(lambda req: pytest.fail("must not call the model")), tmp_path, "replay")
    hit = replay.complete(_req())
    assert hit.cached and hit.text == '{"verdict": "live", "n": 1}'
    assert (tmp_path / "t@1" / f"{request_key(_req())}.json").exists()
    with pytest.raises(LLMUnavailableError, match="no cassette"):
        replay.complete(_req(seed=3))  # different sample -> different key
    filler = CassetteLLM(FakeLLM(lambda req: '{"verdict": "new", "n": 2}'), tmp_path, "record_missing")
    assert filler.complete(_req(seed=3)).text == '{"verdict": "new", "n": 2}' and len(live.calls) == 1


def test_settings_reject_removed_heuristic_knobs():
    with pytest.raises(ValueError, match="was removed"):
        load_settings({"AERO_RETRIEVAL_CREDIBILITY_WEIGHT": "0.1"}, dotenv_path=None)
    with pytest.raises(ValueError, match="unknown grades"):
        load_settings({"AERO_RETRIEVAL_EXCLUDE_RELIABILITY": "E,Z"}, dotenv_path=None)
    s = load_settings({"AERO_RETRIEVAL_EXCLUDE_RELIABILITY": "d, e"}, dotenv_path=None)
    assert s.retrieval_excluded_grades == ("D", "E") and s.judge_model_name == s.llm_model


# ----------------------------------------------------------------- metrics
def test_cohen_kappa_known_values():
    a = ["yes"] * 20 + ["no"] * 15 + ["yes"] * 5 + ["no"] * 10
    b = ["yes"] * 20 + ["no"] * 15 + ["no"] * 5 + ["yes"] * 10
    assert cohen_kappa(a, b) == pytest.approx(0.4, abs=1e-9)  # textbook 2x2 example: po 0.7, pe 0.5
    assert cohen_kappa(["x", "y"], ["x", "y"]) == 1.0 and cohen_kappa(["x", "x"], ["x", "x"]) == 1.0
    levels = ["low", "medium", "high"]
    near = cohen_kappa(["low", "high", "medium"], ["medium", "high", "medium"], labels=levels, weights="quadratic")
    far = cohen_kappa(["low", "high", "medium"], ["high", "high", "medium"], labels=levels, weights="quadratic")
    assert near > far  # an off-by-one disagreement costs less on an ordinal scale
    with pytest.raises(ValueError):
        cohen_kappa(["a"], [])


def test_prf_tau_and_calibration():
    r = prf({1, 2, 3}, {2, 3, 4, 5})
    assert (r.tp, r.fp, r.fn, r.precision, r.recall) == (2, 1, 2, 0.6667, 0.5)
    assert kendall_tau_b([1, 2, 3], [1, 2, 3]) == 1.0 and kendall_tau_b([1, 2, 3], [3, 2, 1]) == -1.0
    assert ranking_tau(["A", "B", "C"], ["A", "C", "B", "D"]) == pytest.approx(1 / 3)
    assert expected_calibration_error([0.9] * 10, [1] * 9 + [0]) == pytest.approx(0.0)
    assert expected_calibration_error([0.9] * 10, [1] * 5 + [0] * 5) == pytest.approx(0.4)
    iso = fit_isotonic([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0, 1, 0, 1, 1, 1])
    assert list(iso.values) == sorted(iso.values) and iso(0.05) == 0.0 and iso(0.9) == 1.0
    assert icd203_term(0.03) == "almost no chance" and icd203_term(0.7) == "likely"
    assert icd203_term(0.5) == "roughly even chance" and icd203_term(0.99) == "almost certain"


# ----------------------------------------------------------------- judgement evaluations
def test_signal_eval_against_gold_with_rule_baseline(corpus_store, taxonomy):
    llm = scripted_llm()
    run_extraction(corpus_store, LLMSignalExtractor(llm, Verifier(llm), taxonomy,
                                                    load_materiality_rubric(SEEDS / "materiality_rubric.yaml")))
    doc = {corpus_store.get_document(i).title: corpus_store.get_document(i) for i in corpus_store.list_document_ids()}
    fcas = doc["Dassault and Airbus clash over FCAS leadership"]
    sirtap = doc["SIRTAP tactical UAS completes first flight"]
    gold = [
        {"canonical_url": fcas.canonical_url, "reviewed_by": "analyst", "signals": [
            {"signal_type": "risk", "topic": "FCAS", "direction": "negative", "event_status": "speculative",
             "materiality": "high"}]},
        {"canonical_url": sirtap.canonical_url, "reviewed_by": "analyst", "signals": [
            {"signal_type": "opportunity", "topic": "Uncrewed Combat Aircraft", "direction": "positive",
             "event_status": "completed", "materiality": "high"}]},
        {"canonical_url": "https://example.com/missing", "reviewed_by": "analyst", "signals": []},
        {"canonical_url": sirtap.canonical_url, "reviewed_by": None, "signals": []},  # draft: ignored
    ]
    baseline = run_baseline(corpus_store, RuleSignalExtractor(load_signal_rules(SEEDS / "signal_rules_v1.yaml"),
                                                              taxonomy))
    rep = jd.evaluate_signals(corpus_store, gold, "signal_extract@1", baseline)
    assert rep["documents"] == 2 and rep["documents_missing_from_db"] == ["https://example.com/missing"]
    assert rep["llm"]["recall"] == 1.0 and rep["matched"] == 2
    assert rep["direction_accuracy"] == 1.0 and rep["event_status_accuracy"] == 1.0
    assert rep["materiality_kappa_quadratic"] < 1.0  # SIRTAP: gold high, model medium
    assert "rule_baseline" in rep and baseline  # the keyword rules still run, for comparison only


def test_sampler_is_stratified_and_reproducible(corpus_store):
    a = jd.sample_signal_documents(corpus_store, 3, seed=1)
    assert a == jd.sample_signal_documents(corpus_store, 3, seed=1)
    assert len(a) == 3 and all(r["reviewed_by"] is None and r["signals"] == [] for r in a)
    assert "FCAS thoughts" not in {r["title"] for r in a}  # quarantined documents are never sampled


def test_judge_and_partner_evals(corpus_store):
    llm = scripted_llm()
    gold = [
        {"claim": "The upgrade was cancelled.", "passage": "The Eurofighter will receive a new suite. More later.",
         "label": "contradicted", "reviewed_by": "analyst"},
        {"claim": "Airbus signed a secure satellite contract with the Bundeswehr.",
         "passage": "Airbus signed a secure satellite communications contract for the Bundeswehr. It is big.",
         "label": "supported", "reviewed_by": "analyst"},
        {"claim": "Saab won a contract.", "passage": "Weather was fine.", "label": "not_enough_info",
         "reviewed_by": "analyst"},
    ]
    rep = jd.evaluate_judge(Verifier(llm), gold)
    assert rep["items"] == 3 and rep["cohen_kappa"] == 1.0 and rep["prompt"] == "verify_claim@1"
    assert jd.evaluate_judge(Verifier(llm), [{**gold[0], "reviewed_by": None}])["items"] == 0
    ranker = PartnerRanker(llm, PartnerRegistry.load(SEEDS / "partners.yaml"), corpus_store, max_workers=2)
    prep = jd.evaluate_partners(ranker, [{"need": "systems integration and sensors", "required_country": "Spain",
                                          "ranking": ["Indra", "GMV"], "reviewed_by": "analyst"}])
    assert prep["mean_tau_b"] == 1.0 and prep["per_need"][0]["system"] == ["Indra", "GMV"]
