"""RAG service — FastAPI entry point.

Endpoints:
  POST /ingest   — ingest a document (semantic chunking + embedding + store)
  POST /query    — agentic RAG query (hybrid search + LangGraph pipeline)
  GET  /health   — service health check
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from rag.chunking import semantic_chunk
from rag.config import CHUNK_BREAKPOINT_THRESHOLD
from rag.graph import run_rag
from rag.retriever import embed_text, ensure_collection, ingest_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rag")

app = FastAPI(title="Agentic RAG Service", version="0.2.0")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    text: str
    title: str = ""
    source: str = ""
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    chunks_stored: int
    chunk_ids: list[str]


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = Field(default_factory=list)
    route: str = ""
    retry_count: int = 0


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    ensure_collection()
    logger.info("RAG service started, collection ensured")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "service": "rag"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Ingest a document: semantic chunk → embed → store in Qdrant."""
    if not req.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text is required.",
        )

    meta = {
        "title": req.title,
        "source": req.source,
        **req.metadata,
    }

    # 1. Semantic chunking
    chunks = semantic_chunk(
        req.text,
        breakpoint_threshold=CHUNK_BREAKPOINT_THRESHOLD,
        metadata=meta,
    )

    # 2. Embed and prepare for storage
    chunk_dicts = []
    chunk_ids = []
    for chunk in chunks:
        cid = str(uuid.uuid4())
        embedding = embed_text(chunk.text)
        chunk_dicts.append(
            {
                "id": cid,
                "text": chunk.text,
                "embedding": embedding,
                "metadata": chunk.metadata,
            }
        )
        chunk_ids.append(cid)

    # 3. Store in Qdrant
    stored = ingest_chunks(chunk_dicts)

    logger.info("Ingested document title='%s' chunks=%d", req.title, stored)
    return IngestResponse(chunks_stored=stored, chunk_ids=chunk_ids)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Run the Agentic RAG pipeline."""
    if not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question is required.",
        )

    result = run_rag(req.question)

    sources = [
        {
            "id": doc.get("id", ""),
            "text": doc.get("text", "")[:200],
            "score": doc.get("score", 0.0),
        }
        for doc in result.get("retrieved_docs", [])
    ]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        route=result.get("route", ""),
        retry_count=result.get("retry_count", 0),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag.app:app", host="0.0.0.0", port=8001, reload=False)
