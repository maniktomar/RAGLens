import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.rag import RAGStore
from app.schemas import ChatRequest, ChatResponse, DocumentSummary, EvaluationRun, QueryLog


load_dotenv()

app = FastAPI(
    title="RAGLens API",
    description="Enterprise RAG chatbot with citation tracing and evaluation metrics.",
    version="0.1.0",
)

origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = RAGStore()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentSummary)
async def upload_document(file: UploadFile = File(...)) -> DocumentSummary:
    try:
        return store.add_document(file.filename or "uploaded.txt", await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    return list(store.documents.values())


@app.delete("/documents")
def clear_documents() -> dict[str, str]:
    store.clear()
    return {"status": "cleared"}


@app.post("/demo/seed", response_model=list[DocumentSummary])
def seed_demo_documents() -> list[DocumentSummary]:
    sample_dir = Path(__file__).resolve().parents[1] / "sample_docs"
    if not sample_dir.exists():
        raise HTTPException(status_code=404, detail="Sample documents folder was not found.")

    summaries: list[DocumentSummary] = []
    for path in sorted(sample_dir.glob("*.txt")):
        summaries.append(store.add_document(path.name, path.read_bytes()))
    return summaries


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    return store.answer(request.question.strip(), request.top_k)


@app.get("/logs", response_model=list[QueryLog])
def logs() -> list[QueryLog]:
    return store.logs


@app.post("/eval/run", response_model=EvaluationRun)
def run_evaluation() -> EvaluationRun:
    if not store.documents:
        seed_demo_documents()
    return store.run_evaluation_suite()
