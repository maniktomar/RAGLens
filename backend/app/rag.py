import csv
import hashlib
import io
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from chromadb import PersistentClient
from chromadb.api.types import EmbeddingFunction
from chromadb.config import Settings
from docx import Document
from openai import OpenAI
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas import AgentStep, ChatResponse, DocumentSummary, Evaluation, EvaluationCase, EvaluationRun, QueryLog, SourceChunk


@dataclass
class Chunk:
    document_id: str
    document_name: str
    chunk_id: str
    text: str


class HashEmbeddingFunction(EmbeddingFunction):
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [hash_embedding(text, self.dimensions) for text in input]


class OpenAIEmbeddingFunction(EmbeddingFunction):
    def __init__(self) -> None:
        self.client = OpenAI()
        self.model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def __call__(self, input: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=input)
        return [item.embedding for item in response.data]


class RAGStore:
    def __init__(self, db_path: str = "data/raglens.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_path = self.db_path.parent / "chroma"
        self.documents: dict[str, DocumentSummary] = {}
        self.document_hashes: dict[str, str] = {}
        self.chunks: list[Chunk] = []
        self.logs: list[QueryLog] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = None
        self.embedding_mode = "openai" if should_use_openai_embeddings() else "local-hash"
        self.embedding_function: EmbeddingFunction = (
            OpenAIEmbeddingFunction() if self.embedding_mode == "openai" else HashEmbeddingFunction()
        )
        self.chroma_client = PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="raglens_chunks",
            embedding_function=self.embedding_function,
            metadata={"description": "RAGLens persistent chunk embeddings"},
        )
        self._init_db()
        self._load_from_db()

    def add_document(self, filename: str, content: bytes) -> DocumentSummary:
        content_hash = hashlib.sha256(content).hexdigest()
        if content_hash in self.document_hashes:
            return self.documents[self.document_hashes[content_hash]]

        text = extract_text(filename, content)
        if not text.strip():
            raise ValueError("No readable text was found in this file.")

        document_id = str(uuid.uuid4())
        new_chunks = [
            Chunk(
                document_id=document_id,
                document_name=filename,
                chunk_id=f"{document_id}:{index}",
                text=chunk,
            )
            for index, chunk in enumerate(chunk_text(text), start=1)
        ]
        self.chunks.extend(new_chunks)
        self._reindex()
        self._index_chroma_chunks(new_chunks)

        summary = DocumentSummary(
            document_id=document_id,
            name=filename,
            chunk_count=len(new_chunks),
            character_count=len(text),
        )
        self.documents[document_id] = summary
        self.document_hashes[content_hash] = document_id
        self._save_document(summary, content_hash, new_chunks)
        return summary

    def clear(self) -> None:
        self.documents.clear()
        self.document_hashes.clear()
        self.chunks.clear()
        self.logs.clear()
        self.matrix = None
        with self._connect() as conn:
            conn.execute("delete from query_logs")
            conn.execute("delete from chunks")
            conn.execute("delete from documents")
        self.chroma_client.delete_collection("raglens_chunks")
        self.collection = self.chroma_client.get_or_create_collection(
            name="raglens_chunks",
            embedding_function=self.embedding_function,
            metadata={"description": "RAGLens persistent chunk embeddings"},
        )

    def search(self, question: str, top_k: int = 5) -> list[SourceChunk]:
        chroma_sources = self._search_chroma(question, top_k)
        if chroma_sources:
            return chroma_sources
        return self._search_tfidf(question, top_k)

    def _search_chroma(self, question: str, top_k: int = 5) -> list[SourceChunk]:
        if not self.chunks:
            return []
        try:
            result = self.collection.query(query_texts=[question], n_results=min(max(top_k * 3, top_k), len(self.chunks)))
        except Exception:
            return []

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        sources: list[SourceChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            base_score = max(1.0 - float(distance), 0.0)
            rerank_bonus = lexical_overlap(question, text) * 0.08
            sources.append(
                SourceChunk(
                    document_id=str(metadata["document_id"]),
                    document_name=str(metadata["document_name"]),
                    chunk_id=str(chunk_id),
                    text=text,
                    score=round(min(base_score + rerank_bonus, 1.0), 4),
                    retrieval_method=f"chroma-{self.embedding_mode}",
                )
            )
        return sorted(sources, key=lambda source: source.score, reverse=True)[:top_k]

    def _search_tfidf(self, question: str, top_k: int = 5) -> list[SourceChunk]:
        if not self.chunks or self.matrix is None:
            return []

        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        candidate_indexes = np.argsort(scores)[::-1][: max(top_k * 3, top_k)]

        sources: list[SourceChunk] = []
        for index in candidate_indexes:
            score = float(scores[index])
            if score <= 0:
                continue
            chunk = self.chunks[index]
            rerank_bonus = lexical_overlap(question, chunk.text) * 0.08
            sources.append(
                SourceChunk(
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    score=round(min(score + rerank_bonus, 1.0), 4),
                    retrieval_method="tfidf-fallback",
                )
            )
        return sorted(sources, key=lambda source: source.score, reverse=True)[:top_k]

    def answer(self, question: str, top_k: int = 5) -> ChatResponse:
        started = time.perf_counter()
        trace = [
            AgentStep(name="Guardrail", status="running", detail="Checking for prompt-injection and unsafe instructions."),
        ]

        guardrail_issue = detect_prompt_injection(question)
        if guardrail_issue:
            trace[0] = AgentStep(name="Guardrail", status="blocked", detail=guardrail_issue)
            latency_ms = int((time.perf_counter() - started) * 1000)
            evaluation = evaluate(question, "I cannot follow instructions that try to override the system or reveal secrets.", [], latency_ms)
            response = ChatResponse(
                answer="I cannot follow instructions that try to override the system or reveal secrets.",
                sources=[],
                evaluation=evaluation,
                agent_trace=trace,
            )
            self._log_response(question, response)
            return response

        trace[0] = AgentStep(name="Guardrail", status="passed", detail="No injection pattern detected.")
        trace.append(AgentStep(name="Retriever", status="running", detail=f"Searching Chroma vector index with {self.embedding_mode} embeddings."))
        sources = self.search(question, top_k)
        method = sources[0].retrieval_method if sources else "none"
        trace[-1] = AgentStep(name="Retriever", status="completed", detail=f"Retrieved {len(sources)} chunks using {method}.")
        trace.append(AgentStep(name="Reranker", status="completed", detail="Reordered candidates with lexical overlap and retrieval score."))

        if not sources:
            answer = "I don't know yet. Upload relevant documents first, or ask about content that exists in the knowledge base."
            trace.append(AgentStep(name="Generator", status="skipped", detail="No context was available for grounded generation."))
        else:
            trace.append(AgentStep(name="Generator", status="running", detail="Generating a grounded answer from cited sources."))
            answer = generate_grounded_answer(question, sources)
            trace[-1] = AgentStep(name="Generator", status="completed", detail="Answer generated with source citations.")

        latency_ms = int((time.perf_counter() - started) * 1000)
        evaluation = evaluate(question, answer, sources, latency_ms)
        trace.append(AgentStep(name="Evaluator", status="completed", detail=f"Confidence {evaluation.confidence:.2f}, risk {evaluation.hallucination_risk:.2f}."))
        response = ChatResponse(answer=answer, sources=sources, evaluation=evaluation, agent_trace=trace)
        self._log_response(question, response)
        return response

    def _reindex(self) -> None:
        corpus = [chunk.text for chunk in self.chunks]
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "create table if not exists documents (document_id text primary key, name text, content_hash text unique, chunk_count integer, character_count integer)"
            )
            conn.execute(
                "create table if not exists chunks (chunk_id text primary key, document_id text, document_name text, text text)"
            )
            conn.execute(
                "create table if not exists query_logs (id integer primary key autoincrement, question text, answer text, created_at text)"
            )

    def _load_from_db(self) -> None:
        with self._connect() as conn:
            for row in conn.execute("select document_id, name, content_hash, chunk_count, character_count from documents"):
                document_id, name, content_hash, chunk_count, character_count = row
                self.documents[document_id] = DocumentSummary(
                    document_id=document_id,
                    name=name,
                    chunk_count=chunk_count,
                    character_count=character_count,
                )
                self.document_hashes[content_hash] = document_id
            for row in conn.execute("select document_id, document_name, chunk_id, text from chunks"):
                self.chunks.append(Chunk(document_id=row[0], document_name=row[1], chunk_id=row[2], text=row[3]))
        self._reindex()
        if self.chunks and self.collection.count() == 0:
            self._index_chroma_chunks(self.chunks)

    def _save_document(self, summary: DocumentSummary, content_hash: str, chunks: list[Chunk]) -> None:
        with self._connect() as conn:
            conn.execute(
                "insert or ignore into documents values (?, ?, ?, ?, ?)",
                (summary.document_id, summary.name, content_hash, summary.chunk_count, summary.character_count),
            )
            conn.executemany(
                "insert or ignore into chunks values (?, ?, ?, ?)",
                [(chunk.chunk_id, chunk.document_id, chunk.document_name, chunk.text) for chunk in chunks],
            )

    def _index_chroma_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        try:
            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                metadatas=[
                    {"document_id": chunk.document_id, "document_name": chunk.document_name}
                    for chunk in chunks
                ],
            )
        except Exception:
            pass

    def _log_response(self, question: str, response: ChatResponse) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        self.logs.insert(
            0,
            QueryLog(
                question=question,
                answer=response.answer,
                sources=response.sources,
                evaluation=response.evaluation,
                created_at=created_at,
                agent_trace=response.agent_trace,
            ),
        )
        self.logs = self.logs[:50]
        with self._connect() as conn:
            conn.execute(
                "insert into query_logs (question, answer, created_at) values (?, ?, ?)",
                (question, response.answer, created_at),
            )

    def run_evaluation_suite(self) -> EvaluationRun:
        cases = [
            ("What is the incident response policy?", "severity 1 incidents require acknowledgement within 15 minutes"),
            ("When should AI-generated customer replies be reviewed?", "billing disputes, security incidents, legal requests, account closures, or customer data deletion"),
            ("What metrics are reviewed for model usage?", "prompt logs, retrieval quality, hallucination reports, latency, estimated cost, and user feedback"),
            ("How quickly must Priority 1 tickets be acknowledged?", "within 10 minutes"),
        ]
        results: list[EvaluationCase] = []
        for question, expected in cases:
            response = self.answer(question, top_k=5)
            score = lexical_overlap(expected, response.answer)
            results.append(
                EvaluationCase(
                    question=question,
                    expected_answer=expected,
                    passed=score >= 0.45,
                    score=round(score, 3),
                    actual_answer=response.answer,
                )
            )
        passed = sum(1 for result in results if result.passed)
        average = float(np.mean([result.score for result in results])) if results else 0.0
        return EvaluationRun(total=len(results), passed=passed, average_score=round(average, 3), cases=results)


def extract_text(filename: str, content: bytes) -> str:
    extension = os.path.splitext(filename.lower())[1]
    if extension == ".pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if extension == ".docx":
        doc = Document(io.BytesIO(content))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    if extension == ".csv":
        frame = pd.read_csv(io.BytesIO(content))
        return frame.to_csv(index=False, quoting=csv.QUOTE_MINIMAL)
    return content.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 140) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= chunk_size:
        return [clean]

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        boundary = clean.rfind(". ", start, end)
        if boundary > start + 250:
            end = boundary + 1
        chunks.append(clean[start:end].strip())
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks


def generate_grounded_answer(question: str, sources: list[SourceChunk]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            return generate_with_openai(question, sources)
        except Exception as exc:
            fallback = generate_extractive_answer(question, sources)
            return (
                f"{fallback}\n\n"
                "OpenAI generation failed, so RAGLens used the local source-grounded fallback. "
                f"Backend detail: {exc.__class__.__name__}."
            )

    return generate_extractive_answer(question, sources)


def generate_extractive_answer(question: str, sources: list[SourceChunk]) -> str:
    bullets = []
    for index, source in enumerate(sources[:3], start=1):
        passage = best_matching_passage(question, source.text)
        bullets.append(f"{index}. {passage} [Source {index}]")

    return (
        "Based on the uploaded knowledge base, the most relevant evidence is:\n\n"
        + "\n".join(bullets)
        + "\n\nAdd an OpenAI API key to enable a more fluent synthesized answer while keeping citations."
    )


def generate_with_openai(question: str, sources: list[SourceChunk]) -> str:
    client = OpenAI()
    context = "\n\n".join(
        f"Source {index} ({source.document_name}): {source.text}"
        for index, source in enumerate(sources, start=1)
    )
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an enterprise RAG assistant. Answer only from the supplied sources. "
                    "If the evidence is insufficient, say you do not know. Cite sources inline as [Source N]."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nSources:\n{context}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or "I don't know based on the uploaded sources."


def best_matching_sentence(question: str, text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    useful_sentences = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 30]
    if not useful_sentences:
        return text[:240]

    question_terms = set(re.findall(r"[a-zA-Z]{4,}", question.lower()))
    if not question_terms:
        return first_useful_sentence(text)

    def score(sentence: str) -> int:
        sentence_terms = set(re.findall(r"[a-zA-Z]{4,}", sentence.lower()))
        return len(question_terms & sentence_terms)

    return max(useful_sentences, key=score)


def best_matching_passage(question: str, text: str) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if len(sentence.strip()) > 30]
    if not sentences:
        return text[:360]

    best = best_matching_sentence(question, text)
    index = sentences.index(best) if best in sentences else 0
    passage = best
    if index + 1 < len(sentences):
        passage = f"{passage} {sentences[index + 1]}"
    return passage[:520]


def first_useful_sentence(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return next((sentence.strip() for sentence in sentences if len(sentence.strip()) > 40), text[:240])


def detect_prompt_injection(question: str) -> str | None:
    patterns = [
        "ignore previous instructions",
        "ignore all instructions",
        "reveal your system prompt",
        "show me your api key",
        "print your secrets",
        "bypass",
    ]
    lowered = question.lower()
    for pattern in patterns:
        if pattern in lowered:
            return f"Blocked prompt-injection pattern: '{pattern}'."
    return None


def should_use_openai_embeddings() -> bool:
    return bool(os.getenv("OPENAI_API_KEY")) and os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"


def hash_embedding(text: str, dimensions: int = 384) -> list[float]:
    vector = np.zeros(dimensions, dtype=np.float32)
    tokens = re.findall(r"[a-zA-Z0-9]{2,}", text.lower())
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()


def lexical_overlap(left: str, right: str) -> float:
    left_terms = set(re.findall(r"[a-zA-Z]{4,}", left.lower()))
    right_terms = set(re.findall(r"[a-zA-Z]{4,}", right.lower()))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


def evaluate(question: str, answer: str, sources: list[SourceChunk], latency_ms: int) -> Evaluation:
    relevance = float(np.mean([source.score for source in sources])) if sources else 0.0
    cited_sources = len(set(re.findall(r"\[Source\s+(\d+)\]", answer)))
    citation_coverage = min(cited_sources / max(len(sources), 1), 1.0) if sources else 0.0
    confidence = min((relevance * 0.7) + (citation_coverage * 0.3), 1.0)
    hallucination_risk = max(1.0 - confidence, 0.0)
    token_estimate = int((len(question) + len(answer) + sum(len(source.text) for source in sources)) / 4)
    estimated_cost_usd = round((token_estimate / 1_000_000) * 0.6, 6)

    return Evaluation(
        confidence=round(confidence, 3),
        citation_coverage=round(citation_coverage, 3),
        hallucination_risk=round(hallucination_risk, 3),
        retrieval_relevance=round(relevance, 3),
        latency_ms=latency_ms,
        estimated_cost_usd=estimated_cost_usd,
        token_estimate=token_estimate,
    )
