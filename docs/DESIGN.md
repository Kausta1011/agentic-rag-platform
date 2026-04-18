# Design Notes

A concise tour of the architectural decisions and the patterns behind them.

## 1. Layering

```
api / mcp / cli / ui      ← transport / presentation
    ↓
agents (LangGraph)        ← orchestration
    ↓
retrieval | tools | guardrails | llm    ← capability layer
    ↓
models (Pydantic)         ← cross-layer contract
    ↓
core (types, errors, logging, config)   ← zero-dependency foundation
```

Every arrow is downward. A lower layer knows nothing about a higher one. This is the Dependency-Inversion leg of SOLID applied at the package level.

## 2. Patterns used (and why)

| Pattern | Where | Why |
|---|---|---|
| **Strategy** | `BaseLLMProvider`, `BaseRetriever`, `BaseTool`, `BaseLoader`, `BaseChunker` | Swap implementations without touching callers |
| **Factory** | `LLMFactory`, `GraphFactory`, `NodeFactory` | Centralise construction; one place to configure |
| **Adapter** | Loaders, MCP server | Hide format-/protocol-specific detail behind a uniform API |
| **Template-Method** | `BaseTool.safe_run` | Exception handling policy is enforced once, per-tool logic is free to do its own thing |
| **Registry** | `ToolRegistry` | Tool discovery without import-time magic |
| **Singleton (cached)** | `get_settings`, `get_llm`, `get_metrics`, `get_service` | Process-wide shared state without mutable globals |
| **Builder-ish DI** | `GraphFactory(retriever=..., reranker=..., tools=...)` | Injectable dependencies for tests |

## 3. Why LangGraph?

We want:

- cycles (reflection loop),
- conditional branching (router / grade),
- observable state transitions,
- streaming,
- a stable surface for checkpointing.

Any of those alone is implementable by hand; all four together is where rolling your own becomes a liability. LangGraph gives all four plus first-class tracing hooks, so every node is trivially a span.

## 4. Why hybrid retrieval + cross-encoder?

BM25 and dense retrieval fail on complementary queries — BM25 on paraphrase, dense on rare tokens and acronyms. Hybrid buys recall. The cross-encoder then buys precision: reading the query and passage *together* with cross-attention is materially more accurate than comparing two independent embeddings.

We run cross-encoder only on the top-N hybrid candidates (`RERANK_TOP_K`) to keep the cost O(N) forward passes, not O(corpus).

## 5. Self-correction (reflection)

After generation, a reflector LLM reviews the draft answer. If it judges the answer insufficient, it proposes a rewritten query and the graph loops back to `rewriter → retrieve → rerank → generate`. Loops are bounded by `MAX_REFLECTION_STEPS` (default 2) so the worst-case latency is bounded.

This is the core trick of *Agentic RAG*: the agent notices when its own output is bad and retries, rather than cheerfully shipping a confident hallucination.

## 6. Guardrails

Two stages:

- **Input**: PII redaction, length cap, prompt-injection heuristics. Fail-closed — raise rather than silently pass through.
- **Output**: faithfulness check via LLM-judge against the retrieved context. If the judge says "unsupported claim", we flag the answer and include the judge's notes in the response metadata.

The faithfulness judge is intentionally the same implementation used offline by the evaluation harness — one grounding check, two callers.

## 7. Observability

- Every graph node opens a span (`tracer.start_as_current_span("node.router")`).
- Every node increments counters and observes a histogram on its latency.
- `/metrics` exposes counters + histogram summaries.

Switching from console to an OTLP collector is one env var: `TRACE_EXPORT=otlp`.

## 8. Testing strategy

- **Unit tests**: fully hermetic via `FakeLLM` and `FakeEmbeddings`.
- **Integration tests**: still hermetic — they wire real BM25 + real nodes, with the fake LLM providing canned JSON so every assertion is deterministic.
- **Eval tests**: the `EvalRunner` is shipped; the repo includes a sample dataset. CI can block a merge on regressions in faithfulness / context precision.

## 9. Things deliberately *not* done

- **No semantic caching** — adds invalidation complexity that belongs behind a real traffic model.
- **No multi-turn memory** — state schema already supports it; a LangGraph checkpointer can be plugged in when a use case appears.
- **No fine-tuning** — the project is about engineering surface, not model training; the LLM layer is provider-agnostic so fine-tuned models slot in as a new concrete provider.

## 10. Where to extend first

1. Swap Chroma for Qdrant (add `src/agentic_rag/retrieval/qdrant_store.py` implementing `BaseRetriever`).
2. Add a HyDE retriever (rewrite query → hypothetical answer → embed → retrieve).
3. Add a moderation classifier to `InputGuard` (replace the heuristic list).
4. Expose the graph over gRPC as well (wrap `service.graph.ainvoke` in a proto service).
5. Add a Prometheus exporter alongside the custom metrics store.
