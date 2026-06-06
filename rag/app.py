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
from rag.watsonx_client import EMBED_DIM, embed_texts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rag")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("QDRANT_COLLECTION", "notes")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", str(EMBED_DIM)))


def _qdrant_headers() -> dict:
    if QDRANT_API_KEY:
        return {"api-key": QDRANT_API_KEY}
    return {}

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


def _llm_error_answer(exc: Exception) -> str | None:
    """Return a user-facing answer for transient watsonx quota/availability errors."""
    message = str(exc)

    if "429" in message or "RESOURCE_EXHAUSTED" in message or "rate" in message.lower():
        return (
            "watsonx API 요청 한도에 걸렸습니다. 잠시 후 다시 시도하거나 "
            "IBM Cloud에서 quota/plan을 확인해주세요."
        )

    if "503" in message or "UNAVAILABLE" in message or "timeout" in message.lower():
        return "watsonx 모델이 일시적으로 혼잡합니다. 잠시 후 다시 시도해주세요."

    return None


# ---------------------------------------------------------------------------
# 컬렉션 자동 생성
# ---------------------------------------------------------------------------

def _ensure_collection():
    try:
        resp = requests.get(
            f"{QDRANT_URL}/collections/{COLLECTION}",
            headers=_qdrant_headers(),
            timeout=5,
        )
        if resp.status_code == 200:
            return
    except requests.RequestException:
        pass
    requests.put(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers=_qdrant_headers(),
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

    # 임베딩 생성 (watsonx) — 청크 전체를 한 번에 배치 처리
    vectors = embed_texts([chunk.text for chunk in chunks])

    points = []
    ids = []
    for chunk, vector in zip(chunks, vectors):
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
        headers=_qdrant_headers(),
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

    try:
        result = rag_graph.invoke({"query": req.question})
    except Exception as exc:
        answer = _llm_error_answer(exc)
        if answer:
            logger.warning("watsonx-backed RAG query failed gracefully: %s", exc)
            return QueryResponse(answer=answer, contexts=[])
        logger.exception("RAG query failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RAG query failed.",
        ) from exc

    return QueryResponse(
        answer=result.get("answer", ""),
        contexts=result.get("contexts", []),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag.app:app", host="0.0.0.0", port=8001, reload=False)
