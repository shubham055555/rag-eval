# Decision Log

## Decision 1 — Establish a Dense Retrieval Baseline

Started with fixed-size chunking and dense retrieval to establish
a measurable baseline.

Result:
49/50 — 98.00% Recall@3.

## Decision 2 — Test Structure-Aware Chunking

Introduced paragraph/section-aware chunking to reduce arbitrary
splits across policy sections.

Result:
50/50 — 100.00% Recall@3.

Decision:
Structure-aware chunking was retained as a strong retrieval
strategy because it achieved the best result on the evaluation set.

## Decision 3 — Test Hybrid Retrieval

Combined BM25 keyword retrieval with dense semantic retrieval
using equal weighting.

Result:
48/50 — 96.00% Recall@3.

Decision:
The initial hybrid configuration was not selected as the standalone
best strategy because it underperformed the dense baseline and
semantic chunking.

## Decision 4 — Add Cross-Encoder Reranking

Used a cross-encoder as a second-stage reranker over retrieved
candidates.

Result:
50/50 — 100.00% Recall@3.

Decision:
Reranking was retained because it recovered the two failures from
the initial hybrid retrieval configuration.
