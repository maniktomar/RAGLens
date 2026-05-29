# RAGLens

Production-style RAG assistant with persistent document indexing, source-grounded answers, agent traceability, guardrails, and an LLM evaluation dashboard.

![Status](https://img.shields.io/badge/status-working_v1-1f7a8c)
![Stack](https://img.shields.io/badge/stack-React_%2B_FastAPI-14213d)
![Focus](https://img.shields.io/badge/focus-GenAI_RAG-e76f51)

## Why This Project Is Recruiter-Friendly

Most GenAI portfolio projects stop at a PDF chatbot. RAGLens adds the parts companies care about in production: persistence, retrieval transparency, answer confidence, citation coverage, hallucination risk, latency, cost visibility, prompt-injection guardrails, and RAG evaluation.

## Features

- Upload `.txt`, `.md`, `.pdf`, `.docx`, and `.csv` knowledge files.
- Chunk and index documents for retrieval.
- Persist documents and chunks in SQLite so the knowledge base survives backend restarts.
- Store chunk vectors in ChromaDB.
- Use deterministic local embeddings by default, with optional OpenAI embeddings for stronger semantic search.
- Ask questions through a polished chat interface.
- Return grounded answers with visible source chunks.
- Show an agent trace for guardrail, retrieval, reranking, generation, and evaluation steps.
- Block common prompt-injection attempts.
- Track confidence, citation coverage, retrieval relevance, hallucination risk, latency, token estimate, and cost estimate.
- Run a built-in RAG evaluation suite against expected answers.
- Keep a query history for auditing.
- Load demo documents with one click.
- Export an answer report as Markdown.
- Works without an API key using extractive answers.
- Uses OpenAI automatically when `OPENAI_API_KEY` is configured.

## Architecture

```text
React UI
  |
  | REST
  v
FastAPI backend
  |
  | parse, chunk, persist, index
  v
SQLite metadata + chunk store
  |
  | ChromaDB vector retrieval + lexical reranking
  v
Agentic RAG pipeline
  |
  | optional generation
  v
OpenAI model
```

The local version uses SQLite for metadata persistence and ChromaDB for persistent vector retrieval. By default, it uses deterministic local embeddings so the project runs without paid APIs. Set `USE_OPENAI_EMBEDDINGS=true` with a valid `OPENAI_API_KEY` to use OpenAI semantic embeddings.

## Run Locally

### Docker Compose

```powershell
cd C:\Users\manik\OneDrive\Documents\Playground\raglens
docker compose up --build
```

Open `http://localhost:8080`.

The backend API is available at `http://localhost:8001`.

Optional OpenAI usage in Docker:

```powershell
$env:OPENAI_API_KEY="your_key_here"
$env:USE_OPENAI_EMBEDDINGS="true"
docker compose up --build
```

Never run `docker compose config` in a public screen recording if API key environment variables are set, because Compose prints resolved environment values.

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload
```

Optional: add `OPENAI_API_KEY` in `backend/.env`.

For OpenAI semantic embeddings:

```env
OPENAI_API_KEY=your_key_here
USE_OPENAI_EMBEDDINGS=true
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Demo Flow

1. Start the backend and frontend.
2. Click `Load sample docs`, or upload files from `backend/sample_docs`.
3. Ask one of these questions:
   - `What is the incident response policy?`
   - `When should AI-generated customer replies be reviewed?`
   - `What metrics are reviewed for model usage?`
4. Inspect the answer, source chunks, and evaluation dashboard.
5. Inspect the agent trace to see guardrail, retrieval, reranking, generation, and evaluation steps.
6. Click `Run RAG eval` to run the built-in evaluation suite.
7. Click `Export` to generate a Markdown answer report.

## Demo Tips

- Use the reset button in the Knowledge Base panel before recording a clean demo.
- Upload at least three sample documents so the dashboard demonstrates retrieval across multiple sources.
- Add an OpenAI key in `backend/.env` for synthesized answers; without a key, the app still works with extractive source-grounded answers.

## Suggested Next Upgrades

- PostgreSQL + pgvector for multi-user production deployment.
- Cross-encoder reranking for stronger retrieval precision.
- Auth and workspace-level document isolation.
- Larger ground-truth evaluation sets.
- PDF report export.
- Docker Compose deployment.
- CI tests for ingestion, retrieval, and answer evaluation.

## Resume Bullets

- Built a full-stack enterprise RAG assistant with FastAPI, React, persistent document indexing, ChromaDB vector retrieval, reranking, source citations, and query history.
- Added an evaluation dashboard that tracks confidence, citation coverage, retrieval relevance, hallucination risk, latency, token estimates, cost, and benchmark pass rate.
- Implemented prompt-injection guardrails and an agent trace showing guardrail, retrieval, reranking, generation, and evaluation stages.
- Designed recruiter-friendly demo workflows with one-click sample data loading and Markdown report export.

## Project Structure

```text
raglens/
  docker-compose.yml
  backend/
    Dockerfile
    app/
      main.py      # FastAPI routes
      rag.py       # parsing, chunking, retrieval, generation, evaluation
      schemas.py   # API response models
    sample_docs/   # demo knowledge base
  frontend/
    Dockerfile
    nginx.conf
    src/
      main.tsx     # React app
      styles.css   # dashboard styling
```
