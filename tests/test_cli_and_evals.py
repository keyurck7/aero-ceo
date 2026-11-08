from __future__ import annotations

import json

import pytest

from aero import cli
from aero.config import get_settings
from aero.evals.retrieval import qrels_from_pool, score_ranking
from tests.conftest import ROOT


def test_score_ranking_metrics():
    rel = {"a": 2, "b": 1, "c": 0}
    r, p, mrr, ndcg = score_ranking(["x", "a", "b"], rel, k=3)
    assert r == 1.0 and p == pytest.approx(2 / 3) and mrr == 0.5
    perfect = score_ranking(["a", "b", "x"], rel, k=3)[3]
    assert perfect == pytest.approx(1.0) and ndcg < perfect
    assert score_ranking([], rel, k=3) == (0.0, 0.0, 0.0, 0.0)
    many = {f"d{i}": 1 for i in range(30)}
    assert score_ranking([f"d{i}" for i in range(10)], many, k=10)[0] == 1.0  # capped recall


def test_qrels_from_pool_keeps_graded_only():
    pool = [{"qid": "q1", "query": "x", "candidates": [{"canonical_url": "u1", "grade": 2},
                                                        {"canonical_url": "u2", "grade": None}]},
            {"qid": "q2", "query": "y", "candidates": [{"canonical_url": "u3", "grade": None}]}]
    assert qrels_from_pool(pool) == [{"qid": "q1", "query": "x", "filters": {}, "relevant": {"u1": 2}}]


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k, v in {
        "AERO_ENV": "test", "AERO_DATABASE_URL": f"sqlite:///{tmp_path / 'cli.sqlite'}",
        "AERO_EMBEDDING_BACKEND": "hashing", "AERO_EMBEDDING_DIM": "256", "AERO_SEEDS_DIR": str(ROOT / "seeds"),
    }.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_cli_end_to_end_v1_import_search_and_eval(cli_env, v1_db, capsys):
    assert cli.main(["db", "migrate"]) == 0
    assert "0001" in capsys.readouterr().out
    assert cli.main(["ingest", "v1", str(v1_db)]) == 0
    imported = json.loads(capsys.readouterr().out)
    assert imported["inserted"] == 3 and imported["stripped_rss_intelligence_note"] == 2
    assert imported["embedded_chunks"] > 0 and imported["run_id"] == 1
    assert cli.main(["db", "stats"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["documents"] == 3 and stats["embedded_chunks"] == stats["chunks"] > 0

    assert cli.main(["search", "SIRTAP first flight", "--json"]) == 0
    hits = json.loads(capsys.readouterr().out)["hits"]
    assert hits[0]["title"] == "SIRTAP drone flies"

    queries = cli_env / "q.jsonl"
    queries.write_text(json.dumps({"qid": "q1", "query": "SIRTAP drone Spain"}) + "\n")
    pool = cli_env / "pool.jsonl"
    assert cli.main(["eval", "pool", "--queries", str(queries), "--out", str(pool), "--k", "5"]) == 0
    rows = [json.loads(line) for line in pool.read_text().splitlines()]
    for cand in rows[0]["candidates"]:  # an analyst would do this by hand
        cand["grade"] = 2 if "SIRTAP" in cand["title"] else 0
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
    qrels = cli_env / "qrels.jsonl"
    assert cli.main(["eval", "labels", "--pool", str(pool), "--out", str(qrels)]) == 0
    capsys.readouterr()
    assert cli.main(["eval", "retrieval", "--qrels", str(qrels), "--k", "3", "--min-recall", "1.0"]) == 0
    assert json.loads(capsys.readouterr().out)["recall@3"] == 1.0
    # The gate fails the build when quality is below the bar.
    impossible = cli_env / "impossible.jsonl"
    impossible.write_text(json.dumps({"qid": "q", "query": "SIRTAP", "relevant": {"https://nowhere/x": 2}}) + "\n")
    assert cli.main(["eval", "retrieval", "--qrels", str(impossible), "--min-recall", "0.5"]) == 1


def test_cli_rejects_bad_config(cli_env, monkeypatch):
    monkeypatch.setenv("AERO_EMBEDING_DIM", "1")  # typo
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="Unknown setting"):
        cli.main(["db", "stats"])


def test_stdio_server_for_desktop_hosts(cli_env, v1_db, capfd):  # capfd: subprocess needs a real stderr fd
    """`aero serve evidence --stdio` is what Claude Desktop / MCP Inspector launch."""
    import os
    import sys

    import anyio
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters

    assert cli.main(["ingest", "v1", str(v1_db)]) == 0
    capfd.readouterr()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    params = StdioServerParameters(command=sys.executable, args=["-m", "aero.cli", "serve", "evidence", "--stdio"],
                                   env=env, cwd=str(cli_env))

    async def go():
        async with Client(params) as c:
            return await c.call_tool("search_evidence", {"query": "Dassault FCAS", "top_k": 1})

    res = anyio.run(go)
    assert not res.is_error and "FCAS" in res.structured_content["hits"][0]["title"]


def test_cli_grading_and_llm_jobs_refuse_without_a_model(cli_env, v1_db, capsys):
    assert cli.main(["ingest", "v1", str(v1_db)]) == 0
    capsys.readouterr()
    assert cli.main(["sources", "grade"]) == 0
    grades = json.loads(capsys.readouterr().out)
    assert sum(grades.values()) >= 1 and set(grades) <= {f"grade_{g}" for g in "ABCDEF"}
    # No AERO_LLM_BASE_URL: every model judgement refuses instead of falling back to a heuristic.
    assert cli.main(["llm", "ping"]) == 1
    assert cli.main(["signals", "extract"]) == 1
    assert "AERO_LLM_BASE_URL" in capsys.readouterr().err
    out = cli_env / "todo.jsonl"
    assert cli.main(["eval", "sample-signals", "--n", "2", "--out", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["reviewed_by"] is None
