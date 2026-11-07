.PHONY: install test test-pg lint up down demo-v1-import serve-stdio eval-pool eval

install:            ## dev install (SQLite + MCP + tests)
	pip install -e ".[mcp,extract,dev]"

test:               ## full suite on SQLite
	pytest

test-pg:            ## contract + retrieval + MCP tests on Postgres (needs AERO_TEST_DATABASE_URL)
	pytest -m postgres -v

lint:
	ruff check src tests scripts/v2
	mypy

up:                 ## Postgres + TEI + evidence-mcp
	docker compose up -d --build

down:
	docker compose down

demo-v1-import:     ## import the v1 corpus into v2 and print stats
	aero db migrate && aero seed sources && aero ingest v1 data/aero_ceo.sqlite && aero db stats

serve-stdio:        ## for Claude Desktop / MCP Inspector
	aero serve evidence --stdio

eval-pool:          ## write candidates to label
	aero eval pool --queries evals/datasets/retrieval_queries.jsonl --out evals/datasets/retrieval_pool.jsonl

eval:               ## score retrieval; fails below the gate
	aero eval labels && aero eval retrieval --min-recall 0.80 --out evals/reports/retrieval_latest.json
