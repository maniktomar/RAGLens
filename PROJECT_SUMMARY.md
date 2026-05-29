# RAGLens Project Summary

## Elevator Pitch

RAGLens is an enterprise-style GenAI knowledge assistant that answers questions from uploaded documents, shows exactly which sources were used, traces the AI workflow, and measures response quality through an evaluation dashboard.

## Problem

Companies want AI assistants, but basic chatbots are difficult to trust because they often lack citations, observability, cost visibility, and hallucination controls.

## Solution

RAGLens combines retrieval-augmented generation with transparent citations, prompt-injection guardrails, persistent indexing, agent traceability, and evaluation metrics. Users can upload documents, ask natural-language questions, inspect source chunks, review confidence and risk signals, run a small benchmark suite, and export an answer report.

## Technical Highlights

- FastAPI backend with document ingestion endpoints.
- Support for text, Markdown, PDF, DOCX, and CSV files.
- SQLite-backed persistence for document metadata and ChromaDB-backed vector storage for chunks.
- Deterministic local embeddings by default, with optional OpenAI embeddings for semantic retrieval.
- Chunking, vector retrieval, TF-IDF fallback, and lexical reranking for local-first demos.
- Optional OpenAI generation when an API key is configured.
- Agentic answer pipeline with guardrail, retriever, reranker, generator, and evaluator steps.
- Built-in RAG evaluation suite with expected-answer scoring.
- React dashboard with knowledge base management, query interface, source viewer, metrics, query history, sample loader, agent trace, RAG eval, and Markdown export.
- Docker Compose setup for reproducible local deployment.

## Why It Matters

This project demonstrates practical GenAI engineering beyond prompt writing: document processing, persistence, retrieval, reranking, grounding, citations, guardrails, evaluation, observability, and user-facing workflow design.

## Best Demo Questions

- What is the incident response policy?
- When should AI-generated customer replies be reviewed?
- What metrics are reviewed for model usage?
- How quickly must Priority 1 tickets be acknowledged?

## Future Production Upgrades

- Replace local ChromaDB with PostgreSQL + pgvector or managed Qdrant for multi-user deployment.
- Add authentication and workspace-level access control.
- Add reranking for higher retrieval precision.
- Add benchmark datasets for automated RAG evaluation.
- Deploy with Docker Compose.
