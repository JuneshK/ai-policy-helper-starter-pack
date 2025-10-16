from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    IngestResponse,
    AskRequest,
    AskResponse,
    MetricsResponse,
    Citation,
    Chunk,
)
from .settings import settings
from .ingest import load_documents
from .rag import RAGEngine, build_chunks_from_docs


# ---------------- FastAPI App ---------------- #
app = FastAPI(title="AI Policy & Product Helper")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Engine
engine = RAGEngine()


# ---------------- Health Check ---------------- #
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------------- Metrics ---------------- #
@app.get("/api/metrics", response_model=MetricsResponse)
def metrics():
    stats = engine.stats()
    return MetricsResponse(**stats)


# ---------------- Ingest Documents ---------------- #
@app.post("/api/ingest", response_model=IngestResponse)
def ingest():
    # Load all docs from data directory
    docs = load_documents(settings.data_dir)

    # Build chunks
    chunks = build_chunks_from_docs(docs, settings.chunk_size, settings.chunk_overlap)

    # Ingest into RAG engine
    new_docs, new_chunks = engine.ingest_chunks(chunks)

    return IngestResponse(indexed_docs=new_docs, indexed_chunks=new_chunks)


# ---------------- Ask / Query ---------------- #
@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest):
    # Retrieve top-k relevant chunks
    context = engine.retrieve(req.query, k=req.k or 4)

    # Generate answer using LLM
    answer = engine.generate(req.query, context)

    # Build citations and chunks for response
    citations = [Citation(title=c.get("title"), section=c.get("section")) for c in context]
    chunks = [Chunk(title=c.get("title"), section=c.get("section"), text=c.get("text")) for c in context]

    # Include metrics
    stats = engine.stats()
    metrics = {
        "retrieval_ms": stats["avg_retrieval_latency_ms"],
        "generation_ms": stats["avg_generation_latency_ms"],
    }

    return AskResponse(
        query=req.query,
        answer=answer,
        citations=citations,
        chunks=chunks,
        metrics=metrics,
    )
