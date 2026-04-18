# Hybrid retrieval, reranking, and fusion

Retrieval-augmented generation lives and dies on retrieval quality. Modern RAG systems fall back on a handful of well-established techniques that together produce substantially better results than any single approach.

## Dense retrieval

A dense retriever embeds the query and every document chunk into a vector space and ranks by cosine similarity. Strengths: semantic matching, paraphrase tolerance, multilingual transfer. Weaknesses: rare tokens, acronyms, exact-string lookups, long-tail entity names.

## BM25

BM25 is a lexical ranking function that rewards exact token overlap and penalises common terms via inverse-document-frequency. It excels precisely where dense retrieval struggles, which is why production systems almost always run it alongside a vector retriever.

## Hybrid fusion

Combining lexical and dense rankings is the cheapest RAG upgrade available. Two popular strategies:

- **Weighted score fusion** — scale each retriever's scores into `[0, 1]` and linearly combine them with a tunable `alpha`. Simple, intuitive, works when the two retrievers' score scales are comparable.
- **Reciprocal rank fusion** (RRF) — ignore the raw scores entirely, use only the ranks: `score = Σ 1 / (k + rank_i)`. RRF is score-free so it is robust to heterogeneous retrievers; the constant `k = 60` from the original paper is rarely worth changing.

## Cross-encoder reranking

After hybrid fusion, the final re-rank is almost always a cross-encoder. Unlike bi-encoders that embed query and passage independently, a cross-encoder reads them together with full cross-attention and outputs a single relevance score. This is O(N) forward passes per query, so you only apply it to the top-N (e.g. N=20) candidates from the hybrid retriever — a two-stage *retrieve → rerank* pipeline.

## Chunking

Chunk size affects retrieval more than most people realise. Too small and you lose context; too large and you dilute the relevance signal and waste prompt budget. A token-budget-aware splitter that respects sentence and paragraph boundaries typically beats naïve character splitting.

## Metrics

Standard RAG metrics include *faithfulness* (is every claim in the answer grounded in the retrieved context?), *answer relevance* (does the answer address the question?), *context precision* (of the retrieved chunks, how many are relevant?), and *context recall* (of the known-relevant chunks, how many did we retrieve?). Track them automatically on every deploy.
