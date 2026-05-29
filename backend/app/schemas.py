from pydantic import BaseModel


class SourceChunk(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    text: str
    score: float
    retrieval_method: str = "unknown"


class AgentStep(BaseModel):
    name: str
    status: str
    detail: str


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5


class Evaluation(BaseModel):
    confidence: float
    citation_coverage: float
    hallucination_risk: float
    retrieval_relevance: float
    latency_ms: int
    estimated_cost_usd: float
    token_estimate: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    evaluation: Evaluation
    agent_trace: list[AgentStep] = []


class DocumentSummary(BaseModel):
    document_id: str
    name: str
    chunk_count: int
    character_count: int


class QueryLog(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    evaluation: Evaluation
    created_at: str
    agent_trace: list[AgentStep] = []


class EvaluationCase(BaseModel):
    question: str
    expected_answer: str
    passed: bool
    score: float
    actual_answer: str


class EvaluationRun(BaseModel):
    total: int
    passed: int
    average_score: float
    cases: list[EvaluationCase]
