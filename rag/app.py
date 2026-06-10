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

from google import genai
from google.genai import errors as genai_errors

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("QDRANT_COLLECTION", "notes")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "3072"))
ES_URL = os.getenv("ES_URL", "")
ES_API_KEY = os.getenv("ES_API_KEY", "")
ES_INDEX = os.getenv("ES_INDEX", "notes")


def _qdrant_headers() -> dict:
    if QDRANT_API_KEY:
        return {"api-key": QDRANT_API_KEY}
    return {}


def _es_headers() -> dict:
    return {
        "Authorization": f"ApiKey {ES_API_KEY}",
        "Content-Type": "application/json",
    }


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
_gemini_client = genai.Client(api_key=GEMINI_API_KEY)

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
    force_retrieve: bool = False


class QueryResponse(BaseModel):
    answer: str
    contexts: list[dict]


def _gemini_error_answer(exc: Exception) -> str | None:
    """Return a user-facing answer for transient Gemini quota/availability errors."""
    status_code = getattr(exc, "status_code", None)
    message = str(exc)

    if isinstance(exc, genai_errors.ClientError) and status_code == 429:
        return (
            "Gemini API 요청 한도에 걸렸습니다. 잠시 후 다시 시도하거나 "
            "Google AI Studio/Google Cloud에서 billing 또는 quota를 확인해주세요."
        )

    if isinstance(exc, genai_errors.ServerError) and status_code == 503:
        return "Gemini 모델이 일시적으로 혼잡합니다. 잠시 후 다시 시도해주세요."

    if "RESOURCE_EXHAUSTED" in message or "429" in message:
        return (
            "Gemini API 요청 한도에 걸렸습니다. 잠시 후 다시 시도하거나 "
            "Google AI Studio/Google Cloud에서 billing 또는 quota를 확인해주세요."
        )

    if "503" in message or "UNAVAILABLE" in message:
        return "Gemini 모델이 일시적으로 혼잡합니다. 잠시 후 다시 시도해주세요."

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


def _ensure_es_index():
    if not ES_URL or not ES_API_KEY:
        logger.info("ES_URL not set, skipping ES index setup")
        return
    try:
        resp = requests.head(
            f"{ES_URL}/{ES_INDEX}",
            headers=_es_headers(),
            timeout=5,
        )
        if resp.status_code == 200:
            logger.info("ES index=%s already exists", ES_INDEX)
            return
    except requests.RequestException:
        pass
    resp = requests.put(
        f"{ES_URL}/{ES_INDEX}",
        headers=_es_headers(),
        json={
            "settings": {
                "analysis": {
                    "analyzer": {
                        "korean": {
                            "type": "nori",
                            "decompound_mode": "mixed",
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text", "analyzer": "korean"},
                    "title": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "timestamp": {"type": "date"},
                }
            },
        },
        timeout=10,
    )
    resp.raise_for_status()
    logger.info("Created ES index=%s with nori analyzer", ES_INDEX)


# ---------------------------------------------------------------------------
# 엔드포인트
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    _ensure_collection()
    _ensure_es_index()


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
        # 임베딩 생성 (Gemini API)
        embed_result = _gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk.text,
        )
        vector = embed_result.embeddings[0].values

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
    logger.info("Ingested %d chunks into Qdrant collection=%s", len(points), COLLECTION)

    # Elasticsearch에 적재 (BM25 키워드 검색용)
    if ES_URL and ES_API_KEY:
        bulk_body = ""
        for p in points:
            bulk_body += f'{{"index": {{"_index": "{ES_INDEX}", "_id": "{p["id"]}"}}}}\n'
            doc = {k: v for k, v in p["payload"].items()}
            import json as _json
            bulk_body += _json.dumps(doc, ensure_ascii=False) + "\n"
        es_resp = requests.post(
            f"{ES_URL}/_bulk",
            headers=_es_headers(),
            data=bulk_body.encode("utf-8"),
            timeout=30,
        )
        if es_resp.ok:
            logger.info("Ingested %d chunks into ES index=%s", len(points), ES_INDEX)
        else:
            logger.warning("ES bulk ingest failed: %s %s", es_resp.status_code, es_resp.text[:200])

    return IngestResponse(chunks_stored=len(points), ids=ids)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """LangGraph Agentic RAG 파이프라인으로 질의 처리."""
    if not req.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question is empty.")

    try:
        result = rag_graph.invoke({"query": req.question, "force_retrieve": req.force_retrieve})
    except Exception as exc:
        answer = _gemini_error_answer(exc)
        if answer:
            logger.warning("Gemini-backed RAG query failed gracefully: %s", exc)
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
