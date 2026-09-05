# RAG Evaluation & Hardness

An evaluation-driven Retrieval-Augmented Generation (RAG) system that compares multiple retrieval strategies using a fixed 50-question benchmark.

## Overview

This project focuses on measuring and improving retrieval quality rather than building a generic RAG chatbot.

The system evaluates:

- Fixed-size dense retrieval
- Structure-aware semantic-style chunking
- Hybrid BM25 + dense retrieval
- Hybrid retrieval with cross-encoder reranking

A GitHub Actions quality gate automatically fails the build if Recall@3 drops below 95%.

## Evaluation Results

| Retrieval Strategy | Correct | Recall@3 |
|---|---:|---:|
| Fixed-size Dense Retrieval | 49/50 | 98% |
| Structure-aware Chunking | 50/50 | 100% |
| Hybrid BM25 + Dense | 48/50 | 96% |
| Hybrid + Cross-Encoder Reranking | 50/50 | 100% |


## Key Finding

The best-performing configuration was:

**Structure-aware chunking + hybrid retrieval + cross-encoder reranking**

Reranking recovered the failures introduced by the equal-weight hybrid approach and achieved 100% Recall@3 on the 50-question benchmark.

## Architecture

Documents
    |
    v
Chunking
    |
    v
Dense Retrieval + BM25
    |
    v
Hybrid Candidate Retrieval
    |
    v
Cross-Encoder Reranking
    |
    v
Top-K Context
    |
    v
Evaluation Harness
    |
    v
Recall@3 + Failure Analysis
    |
    v
GitHub Actions Quality Gate

## Engineering Decisions

### Fixed-size chunking

Used as the initial baseline because it provides a simple and measurable reference point.

### Structure-aware chunking

Paragraph-aware splitting improved retrieval from 98% to 100% Recall@3.

### Hybrid retrieval

BM25 was combined with dense retrieval to support both lexical and semantic matching. The equal-weight configuration achieved 96% Recall@3 on this benchmark.

### Cross-encoder reranking

A cross-encoder was used to rerank the top candidate documents. This recovered the hybrid retrieval failures and achieved 100% Recall@3.

## Failure Analysis

The hybrid configuration failed two benchmark questions:

- Q40 — expected source: `code_of_conduct.txt`
- Q47 — expected source: `performance_policy.txt`

Cross-encoder reranking recovered both failures.

## Limitations

- The benchmark contains 50 questions.
- Evaluation focuses on retrieval quality rather than end-to-end answer faithfulness.
- The current semantic chunker is structure-aware rather than embedding-based breakpoint detection.
- Results are benchmark-specific and should not be treated as universal retrieval performance.

## Future Improvements

- Add answer-level evaluation.
- Add faithfulness and citation metrics.
- Experiment with different embedding models.
- Tune hybrid retrieval weights automatically.
- Expand the evaluation dataset.
- Add experiment tracking and regression dashboards.

