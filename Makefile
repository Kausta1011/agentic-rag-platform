# ---- Agentic RAG Platform — Makefile --------------------------------------
.PHONY: help install dev fmt lint test test-fast cov smoke ingest eval api ui mcp docker docker-up docker-down clean

PY ?= python

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Install package (editable)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e .

dev:  ## Install package + dev extras
	$(PY) -m pip install -e '.[dev]'

fmt:  ## Ruff format
	$(PY) -m ruff format src tests scripts ui

lint:  ## Ruff lint
	$(PY) -m ruff check src tests scripts

test:  ## Full test suite with coverage
	$(PY) -m pytest

test-fast:  ## Unit tests only, no coverage
	$(PY) -m pytest tests/unit -q --no-cov

cov:  ## Coverage HTML report
	$(PY) -m pytest --cov-report=html
	@echo "HTML report: htmlcov/index.html"

smoke:  ## Import-only smoke check
	$(PY) -c "import agentic_rag, agentic_rag.api, agentic_rag.agents, agentic_rag.retrieval; print('ok')"

ingest:  ## Ingest the sample corpus
	$(PY) -m scripts.ingest --path data/corpus

eval:  ## Run the sample evaluation
	$(PY) -m scripts.evaluate --dataset tests/eval/sample_dataset.json --out reports/eval.json

api:  ## Start the FastAPI server
	uvicorn agentic_rag.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:  ## Start the Streamlit UI
	streamlit run ui/streamlit_app.py

mcp:  ## Start the MCP server (stdio)
	$(PY) -m mcp_server.server

docker:  ## Build the docker image
	docker build -t agentic-rag-platform:latest .

docker-up:  ## Start api + ui via docker-compose
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov build dist *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
