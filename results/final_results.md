# RAG Evaluation Summary

## Evaluation Setup

- Documents: 8
- Evaluation questions: 50
- Metric: Recall@3
- Embedding model: all-MiniLM-L6-v2

## Results

| Strategy | Correct | Recall@3 |
|---|---:|---:|
| Fixed-size Dense Retrieval | 49/50 | 98.00% |
| Structure-aware Semantic Chunking | 50/50 | 100.00% |
| Hybrid BM25 + Dense | 48/50 | 96.00% |
| Hybrid + Cross-Encoder Reranking | 50/50 | 100.00% |

## Findings

Structure-aware semantic chunking achieved the highest retrieval
performance on the evaluation set.

The equal-weight hybrid configuration achieved 96.00% Recall@3,
which was lower than both fixed-size dense retrieval and
structure-aware semantic chunking.

Adding cross-encoder reranking recovered the hybrid pipeline's
retrieval quality to 100.00% Recall@3.

## Failure Analysis

The hybrid configuration produced two failures:

- Q40: expected `code_of_conduct.txt`
- Q47: expected `performance_policy.txt`

The reranking stage successfully recovered both expected sources
within the final Top-3 results.

## Engineering Decision

The experiments show that combining retrieval techniques does not
automatically improve retrieval quality. Chunking strategy, score
normalization, retrieval weights, and second-stage reranking all
affect final performance.

For this evaluation set, structure-aware semantic retrieval and
hybrid retrieval with cross-encoder reranking both achieved
100.00% Recall@3.
