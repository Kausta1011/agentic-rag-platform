# Agentic RAG Platform

Production-grade **Agentic Retrieval-Augmented Generation** system built with **LangGraph**, hybrid retrieval, cross-encoder reranking, self-correction, guardrails, evaluation harness, observability, FastAPI, and an MCP server — the full 2026 AI-engineer stack, end-to-end.

> Built as a portfolio piece targeting **LLM / GenAI Engineer** roles. The project's focus is not on "a RAG that works" — every RAG works on the happy path — but on the *engineering surface* that distinguishes a demo from a production system: evaluation, observability, guardrails, typed contracts, pluggable providers, clean LLD, and first-class MCP integration.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Streamlit UI  /  MCP Client                    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ HTTP + SSE  /  stdio
┌──────────────────────────────▼───────────────────────────────────────┐
│                    FastAPI backend  +  MCP server                    │
│           (query / query/stream / metrics / healthz)                 │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                   ┌───────────▼──────────┐   InputGuard (PII, length,
                   │   LangGraph agent    │◄──  prompt-injection)
                   └───────────┬──────────┘
                               │
       ┌───────────────────────┼────────────────────────────────┐
       │                       │                                │
  [router] ───► [rewriter] ───►[retrieve]───►[rerank]───►[grade]
       │                                                        │
       │           web_search ◄──────────────────────── (irrelevant)
       │                │
       └───► [generate] ◄── (relevant)
                    │
                 [reflect] ── loop (≤ MAX_REFLECTION_STEPS) ──► rewriter
                    │
                    ▼
               OutputGuard (faithfulness)
                    │
                    ▼
                  client
```

## Key features

| Layer | What's in it | Why it matters |
|---|---|---|
| **Orchestration** | LangGraph state-machine with router, rewriter, retrieve, rerank, grade, web-search, generate, reflect | Stateful, branchable, resumable |
| **Retrieval** | Chroma dense + BM25 lexical + weighted / RRF fusion + cross-encoder rerank | Hybrid recovers lexical queries; reranker lifts precision |
| **LLM layer** | Strategy + Factory over OpenAI & Anthropic; streaming; normalised `LLMResponse` | Provider swap is zero-change for callers |
| **Ingestion** | Text / Markdown / PDF / Web / Directory loaders; token-aware chunker; preprocessor | Adapter pattern, streams, idempotent chunk IDs |
| **Tools** | Web search (Tavily → DDG fallback); sandboxed calculator; tool registry | Safe tool execution, graceful failure |
| **Guardrails** | Input (PII redaction, injection heuristics, length cap) + Output (LLM-judged faithfulness) | Fail-closed defaults |
| **Observability** | OpenTelemetry spans per node; in-process metrics; `/metrics` endpoint | Every node call is traceable |
| **Evaluation** | RAGAS-style faithfulness, answer relevance, context precision/recall; CLI runner | Measurable quality over time |
| **API** | FastAPI + SSE + Pydantic contracts + DI + CORS | Production-ready boilerplate |
| **MCP** | Stdio MCP server exposing `ask_corpus` & `ingest_document` tools | 2026-era tool interop |
| **UI** | Streamlit chat with node-stream view | Honest demo, thin client |
| **Testing** | Unit + integration + fake LLM / embeddings; `pytest-asyncio` | Hermetic, fast, deterministic |
| **Deployment** | Dockerfile, docker-compose (api + ui), Makefile, pyproject | One command to ship |

## Quick start

```bash
# 1) Clone / open the project
cd agentic-rag-platform

# 2) Install deps
make dev

# 3) Fill in API keys
#   (the repo stores env vars in a *folder* called .env;
#    edit .env/variables.env and set at least ANTHROPIC_API_KEY
#    or OPENAI_API_KEY)
$EDITOR .env/variables.env

# 4) Ingest the sample corpus
make ingest

# 5) Start the API
make api        # http://localhost:8000/docs

# 6) In another shell, start the UI
make ui         # http://localhost:8501
```

Or, all-in with Docker:

```bash
make docker-up  # api on :8000, ui on :8501
```

## Configuration

All config is loaded from env variables via `pydantic-settings`. The repository stores env files in a **folder** (`.env/`) — each file inside it is loaded on import, so you can split concerns (`openai.env`, `anthropic.env`, `infra.env`, …).

See [`.env/variables.env`](./.env/variables.env) for every knob (LLM provider, embedding provider, retrieval top-k, hybrid alpha, reflection depth, guardrail thresholds, OTLP endpoint, …).

## Evaluation

```bash
make eval       # runs the sample dataset and writes reports/eval.json
```

The harness computes **faithfulness**, **answer relevance**, **context precision**, and **context recall** per case, plus a summary with average and percentile latency.

## MCP server

```bash
make mcp        # stdio transport
```

Point Claude Desktop / Cursor / Cowork / any MCP-aware client at this binary and you get two tools: `ask_corpus` (full pipeline) and `ingest_document`.

## Design principles

- **SOLID**: every module has a single responsibility; consumers depend on abstractions (`BaseLLMProvider`, `BaseRetriever`, `BaseTool`, `BaseLoader`, `BaseChunker`).
- **Factory / Strategy / Template-Method**: concrete providers, retrievers, tools, chunkers all live behind factories.
- **Contract-first**: Pydantic models are the only cross-boundary interchange.
- **Fail-closed**: guardrails default to on; tool failures are captured as `ToolResult(ok=False, …)`, not exceptions; LLM failures become typed `LLMProviderError`.
- **Observability before cleverness**: every node is a traced span; every call is a metric.
- **Hermetic tests**: `FakeLLM` and `FakeEmbeddings` make every unit test deterministic.

## Project layout

```
agentic-rag-platform/
├── .env/                       # env var folder (gitignored)
├── src/agentic_rag/
│   ├── config.py               # pydantic-settings; single source of truth
│   ├── core/                   # logging, typed enums, exception hierarchy
│   ├── models/                 # Document, Chunk, ScoredChunk, AgentState, I/O
│   ├── llm/                    # Strategy + Factory + (OpenAI | Anthropic)
│   ├── retrieval/              # chunking, vector store, BM25, hybrid, reranker
│   ├── ingestion/              # loaders + preprocessor + pipeline
│   ├── tools/                  # web_search, calculator + registry
│   ├── agents/                 # LangGraph nodes, prompts, graph factory
│   ├── guardrails/             # input + output guards
│   ├── observability/          # tracer + metrics
│   ├── evaluation/             # dataset, metrics, runner
│   └── api/                    # FastAPI app, routes, DI, SSE streaming
├── src/mcp_server/             # Model Context Protocol server
├── ui/streamlit_app.py         # Demo UI
├── scripts/                    # ingest + evaluate CLIs
├── tests/                      # unit + integration + eval dataset
├── data/corpus/                # sample documents
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Roadmap / "if I had another week"

- Multi-query decomposition for complex questions
- Persistent conversation memory via LangGraph checkpointer
- Vector-store swap to Qdrant / pgvector
- Prompt-based jailbreak classifier in `InputGuard`
- Prometheus exporter in `observability.metrics`
- HyDE (Hypothetical Document Embeddings) retriever variant
- CI: GitHub Actions with ruff + pytest + docker build

## License

MIT — see `pyproject.toml`.
