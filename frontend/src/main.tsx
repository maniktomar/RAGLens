import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Bot,
  CheckCircle2,
  Database,
  Download,
  FileSearch,
  Gauge,
  Loader2,
  RotateCcw,
  Send,
  ShieldCheck,
  TestTube2,
  Upload,
} from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type SourceChunk = {
  document_id: string;
  document_name: string;
  chunk_id: string;
  text: string;
  score: number;
  retrieval_method: string;
};

type Evaluation = {
  confidence: number;
  citation_coverage: number;
  hallucination_risk: number;
  retrieval_relevance: number;
  latency_ms: number;
  estimated_cost_usd: number;
  token_estimate: number;
};

type AgentStep = {
  name: string;
  status: string;
  detail: string;
};

type ChatResponse = {
  answer: string;
  sources: SourceChunk[];
  evaluation: Evaluation;
  agent_trace: AgentStep[];
};

type DocumentSummary = {
  document_id: string;
  name: string;
  chunk_count: number;
  character_count: number;
};

type QueryLog = ChatResponse & {
  question: string;
  created_at: string;
};

type EvaluationRun = {
  total: number;
  passed: number;
  average_score: number;
  cases: {
    question: string;
    expected_answer: string;
    passed: boolean;
    score: number;
    actual_answer: string;
  }[];
};

function App() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [logs, setLogs] = useState<QueryLog[]>([]);
  const [question, setQuestion] = useState("What is the incident response policy?");
  const [activeAnswer, setActiveAnswer] = useState<ChatResponse | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [isSeeding, setIsSeeding] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalRun, setEvalRun] = useState<EvaluationRun | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    refresh();
  }, []);

  const dashboard = useMemo(() => {
    const latest = activeAnswer?.evaluation ?? logs[0]?.evaluation;
    return {
      documents: documents.length,
      chunks: documents.reduce((sum, doc) => sum + doc.chunk_count, 0),
      confidence: latest?.confidence ?? 0,
      hallucinationRisk: latest?.hallucination_risk ?? 0,
      latency: latest?.latency_ms ?? 0,
    };
  }, [activeAnswer, documents, logs]);

  async function refresh() {
    const [documentResponse, logResponse] = await Promise.all([
      fetch(`${API_URL}/documents`),
      fetch(`${API_URL}/logs`),
    ]);
    setDocuments(await documentResponse.json());
    setLogs(await logResponse.json());
  }

  async function uploadFile(file: File) {
    setIsUploading(true);
    setError("");
    const form = new FormData();
    form.append("file", file);
    const response = await fetch(`${API_URL}/documents`, {
      method: "POST",
      body: form,
    });
    setIsUploading(false);

    if (!response.ok) {
      setError((await response.json()).detail ?? "Upload failed.");
      return;
    }
    await refresh();
  }

  async function askQuestion(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim()) return;

    setIsAsking(true);
    setError("");
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: 5 }),
    });
    setIsAsking(false);

    if (!response.ok) {
      setError((await response.json()).detail ?? "Question failed.");
      return;
    }

    const result = await response.json();
    setActiveAnswer(result);
    await refresh();
  }

  async function clearKnowledgeBase() {
    setError("");
    await fetch(`${API_URL}/documents`, { method: "DELETE" });
    setActiveAnswer(null);
    await refresh();
  }

  async function loadSamples() {
    setIsSeeding(true);
    setError("");
    const response = await fetch(`${API_URL}/demo/seed`, { method: "POST" });
    setIsSeeding(false);
    if (!response.ok) {
      setError((await response.json()).detail ?? "Could not load sample documents.");
      return;
    }
    await refresh();
  }

  function exportReport() {
    if (!activeAnswer) return;
    const lines = [
      "# RAGLens Answer Report",
      "",
      `**Question:** ${question}`,
      "",
      "## Answer",
      "",
      activeAnswer.answer,
      "",
      "## Evaluation",
      "",
      `- Confidence: ${percent(activeAnswer.evaluation.confidence)}`,
      `- Citation coverage: ${percent(activeAnswer.evaluation.citation_coverage)}`,
      `- Retrieval relevance: ${percent(activeAnswer.evaluation.retrieval_relevance)}`,
      `- Hallucination risk: ${percent(activeAnswer.evaluation.hallucination_risk)}`,
      `- Latency: ${activeAnswer.evaluation.latency_ms}ms`,
      `- Estimated cost: $${activeAnswer.evaluation.estimated_cost_usd.toFixed(6)}`,
      "",
      "## Sources",
      "",
      ...activeAnswer.sources.flatMap((source, index) => [
        `### Source ${index + 1}: ${source.document_name}`,
        "",
        `Relevance: ${percent(source.score)}`,
        "",
        source.text,
        "",
      ]),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "raglens-answer-report.md";
    link.click();
    URL.revokeObjectURL(url);
  }

  async function runEvaluation() {
    setIsEvaluating(true);
    setError("");
    const response = await fetch(`${API_URL}/eval/run`, { method: "POST" });
    setIsEvaluating(false);
    if (!response.ok) {
      setError((await response.json()).detail ?? "Evaluation run failed.");
      return;
    }
    setEvalRun(await response.json());
    await refresh();
  }

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandMark">
            <FileSearch size={22} />
          </div>
          <div>
            <strong>RAGLens</strong>
            <span>Auditable enterprise AI</span>
          </div>
        </div>

        <label className="uploadZone">
          {isUploading ? <Loader2 className="spin" /> : <Upload />}
          <span>{isUploading ? "Indexing..." : "Upload knowledge"}</span>
          <input
            type="file"
            accept=".txt,.md,.pdf,.docx,.csv"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) uploadFile(file);
            }}
          />
        </label>

        <button className="secondaryAction" onClick={loadSamples} disabled={isSeeding}>
          {isSeeding ? <Loader2 className="spin" /> : <Database size={17} />}
          <span>{isSeeding ? "Loading samples..." : "Load sample docs"}</span>
        </button>

        <button className="secondaryAction" onClick={runEvaluation} disabled={isEvaluating}>
          {isEvaluating ? <Loader2 className="spin" /> : <TestTube2 size={17} />}
          <span>{isEvaluating ? "Running eval..." : "Run RAG eval"}</span>
        </button>

        <section className="panel compact">
          <div className="panelHeader">
            <Database size={18} />
            <h2>Knowledge Base</h2>
            {documents.length > 0 && (
              <button className="iconButton" title="Clear knowledge base" onClick={clearKnowledgeBase}>
                <RotateCcw size={16} />
              </button>
            )}
          </div>
          {documents.length === 0 ? (
            <p className="muted">Upload documents or use the sample docs in the backend folder.</p>
          ) : (
            <div className="docList">
              {documents.map((doc) => (
                <div className="docRow" key={doc.document_id}>
                  <span>{doc.name}</span>
                  <small>{doc.chunk_count} chunks</small>
                </div>
              ))}
            </div>
          )}
        </section>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>Enterprise RAG Chatbot</h1>
            <p>Ask grounded questions, inspect citations, and measure answer quality.</p>
          </div>
          <div className="statusPill">
            <ShieldCheck size={16} />
            Source-grounded
          </div>
        </header>

        {error && <div className="error">{error}</div>}

        <section className="metricGrid">
          <Metric icon={<Database />} label="Documents" value={String(dashboard.documents)} />
          <Metric icon={<FileSearch />} label="Chunks" value={String(dashboard.chunks)} />
          <Metric icon={<Gauge />} label="Confidence" value={percent(dashboard.confidence)} />
          <Metric icon={<Activity />} label="Risk" value={percent(dashboard.hallucinationRisk)} />
          <Metric icon={<Bot />} label="Latency" value={`${dashboard.latency}ms`} />
        </section>

        <section className="mainGrid">
          <div className="chatPanel">
            <div className="panelHeader">
              <Bot size={19} />
              <h2>Ask The Knowledge Base</h2>
            </div>
            <form className="askBar" onSubmit={askQuestion}>
              <input value={question} onChange={(event) => setQuestion(event.target.value)} />
              <button type="button" className="ghostButton" disabled={!activeAnswer} onClick={exportReport}>
                <Download size={17} />
                <span>Export</span>
              </button>
              <button disabled={isAsking}>
                {isAsking ? <Loader2 className="spin" /> : <Send size={17} />}
                <span>Ask</span>
              </button>
            </form>

            <article className="answerBox">
              {activeAnswer ? (
                <>
                  <pre>{activeAnswer.answer}</pre>
                  <div className="traceGrid">
                    {activeAnswer.agent_trace.map((step) => (
                      <div className="traceStep" key={`${step.name}-${step.status}`}>
                        <CheckCircle2 size={16} />
                        <div>
                          <strong>{step.name}</strong>
                          <span>{step.status}</span>
                          <p>{step.detail}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="sourceGrid">
                    {activeAnswer.sources.map((source, index) => (
                      <div className="sourceCard" key={source.chunk_id}>
                        <div>
                          <strong>Source {index + 1}</strong>
                          <span>{source.document_name}</span>
                        </div>
                        <small>
                          Relevance {percent(source.score)} · {source.retrieval_method}
                        </small>
                        <p>{source.text}</p>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="emptyState">
                  <FileSearch size={42} />
                  <h2>Upload documents, then ask a question.</h2>
                  <p>The assistant will show cited answers and evaluation signals for every query.</p>
                </div>
              )}
            </article>
          </div>

          <aside className="evalPanel">
            <div className="panelHeader">
              <Gauge size={19} />
              <h2>Evaluation Dashboard</h2>
            </div>
            <EvalBars evaluation={activeAnswer?.evaluation ?? logs[0]?.evaluation} />
            <div className="history">
              <h3>Recent Queries</h3>
              {logs.slice(0, 5).map((log) => (
                <button key={`${log.created_at}-${log.question}`} onClick={() => setActiveAnswer(log)}>
                  <span>{log.question}</span>
                  <small>{percent(log.evaluation.confidence)} confidence</small>
                </button>
              ))}
            </div>
            {evalRun && (
              <div className="evalSuite">
                <h3>RAG Eval Suite</h3>
                <div className="scoreBadge">
                  <strong>
                    {evalRun.passed}/{evalRun.total}
                  </strong>
                  <span>{percent(evalRun.average_score)} avg score</span>
                </div>
                {evalRun.cases.map((testCase) => (
                  <div className="caseRow" key={testCase.question}>
                    <strong>{testCase.passed ? "Pass" : "Review"}</strong>
                    <span>{testCase.question}</span>
                    <small>{percent(testCase.score)}</small>
                  </div>
                ))}
              </div>
            )}
          </aside>
        </section>
      </section>
    </main>
  );
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return (
    <div className="metric">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EvalBars({ evaluation }: { evaluation?: Evaluation }) {
  if (!evaluation) {
    return <p className="muted">Evaluation metrics appear after the first answer.</p>;
  }
  return (
    <div className="bars">
      <Bar label="Confidence" value={evaluation.confidence} />
      <Bar label="Citation coverage" value={evaluation.citation_coverage} />
      <Bar label="Retrieval relevance" value={evaluation.retrieval_relevance} />
      <Bar label="Hallucination risk" value={evaluation.hallucination_risk} danger />
      <div className="costBox">
        <span>{evaluation.token_estimate} tokens estimated</span>
        <strong>${evaluation.estimated_cost_usd.toFixed(6)}</strong>
      </div>
    </div>
  );
}

function Bar({ label, value, danger = false }: { label: string; value: number; danger?: boolean }) {
  return (
    <div className="barRow">
      <div>
        <span>{label}</span>
        <strong>{percent(value)}</strong>
      </div>
      <div className="barTrack">
        <div className={danger ? "bar danger" : "bar"} style={{ width: percent(value) }} />
      </div>
    </div>
  );
}

function percent(value: number) {
  return `${Math.round(value * 100)}%`;
}

createRoot(document.getElementById("root")!).render(<App />);
