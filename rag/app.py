"""RAG 서비스 FastAPI 엔드포인트."""

import logging
import os
import uuid
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from rag.chunking import semantic_chunk
from rag.graph import rag_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rag")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION = os.getenv("QDRANT_COLLECTION", "notes")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))

app = FastAPI(title="RAG Service", version="0.1.0")


# ---------------------------------------------------------------------------
# 요청/응답 모델
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    text: str
    title: str = ""
    metadata: dict = {}


class IngestResponse(BaseModel):
    chunks_stored: int
    ids: list[str]


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    contexts: list[dict]


# ---------------------------------------------------------------------------
# 컬렉션 자동 생성
# ---------------------------------------------------------------------------

def _ensure_collection():
    try:
        resp = requests.get(f"{QDRANT_URL}/collections/{COLLECTION}", timeout=5)
        if resp.status_code == 200:
            return
    except requests.RequestException:
        pass
    requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        json={
            "vectors": {"size": VECTOR_SIZE, "distance": "Cosine"},
        },
        timeout=10,
    ).raise_for_status()
    logger.info("Created Qdrant collection=%s size=%d", COLLECTION, VECTOR_SIZE)


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    _ensure_collection()


@app.get("/health")
async def health():
    return {"status": "ok", "collection": COLLECTION}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """텍스트를 Semantic Chunking → 임베딩 → Qdrant에 저장."""
    if not req.text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text is empty.")

    chunks = semantic_chunk(req.text)
    if not chunks:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No chunks produced.")

    points = []
    ids = []
    for chunk in chunks:
        # 임베딩 생성
        embed_resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": chunk.text},
            timeout=30,
        )
        embed_resp.raise_for_status()
        vector = embed_resp.json()["embedding"]

        point_id = str(uuid.uuid4())
        ids.append(point_id)
        points.append({
            "id": point_id,
            "vector": vector,
            "payload": {
                "text": chunk.text,
                "chunk_index": chunk.index,
                "title": req.title,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **req.metadata,
            },
        })

    # Qdrant에 적재
    resp = requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}/points",
        json={"points": points},
        timeout=30,
    )
    resp.raise_for_status()
    logger.info("Ingested %d chunks into collection=%s", len(points), COLLECTION)

    return IngestResponse(chunks_stored=len(points), ids=ids)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """LangGraph Agentic RAG 파이프라인으로 질의 처리."""
    if not req.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question is empty.")

    result = rag_graph.invoke({"query": req.question})
    return QueryResponse(
        answer=result.get("answer", ""),
        contexts=result.get("contexts", []),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag.app:app", host="0.0.0.0", port=8001, reload=False)
