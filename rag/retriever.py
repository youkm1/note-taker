"""Hybrid Search (Elasticsearch nori BM25 + 벡터) → RRF 융합 → Cross-Encoder Reranker."""

import logging
import os
from dataclasses import dataclass, field

import numpy as np
import requests
from sentence_transformers import CrossEncoder

logger = logging.getLogger("rag.retriever")

from google import genai

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION = os.getenv("QDRANT_COLLECTION", "notes")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
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


_gemini_client = genai.Client(api_key=GEMINI_API_KEY)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = CrossEncoder(RERANKER_MODEL)


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Elasticsearch BM25 (nori 형태소 분석기)
# ---------------------------------------------------------------------------

def _es_search(query: str, top_k: int = 20) -> list[SearchResult]:
    if not ES_URL or not ES_API_KEY:
        logger.warning("ES_URL or ES_API_KEY not set, skipping ES search")
        return []
    try:
        resp = requests.post(
            f"{ES_URL}/{ES_INDEX}/_search",
            headers=_es_headers(),
            json={
                "query": {
                    "match": {
                        "text": {
                            "query": query,
                        }
                    }
                },
                "size": top_k,
            },
            timeout=10,
        )
        resp.raise_for_status()
        results = []
        for hit in resp.json()["hits"]["hits"]:
            results.append(SearchResult(
                id=hit["_id"],
                text=hit["_source"].get("text", ""),
                score=hit["_score"] or 0.0,
                metadata=hit["_source"],
            ))
        return results
    except Exception as e:
        logger.warning("ES search failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# 벡터 검색
# ---------------------------------------------------------------------------

def _embed_query(text: str) -> list[float]:
    result = _gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return result.embeddings[0].values


def _vector_search(query: str, top_k: int = 20) -> list[SearchResult]:
    vector = _embed_query(query)
    resp = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
        headers=_qdrant_headers(),
        json={"vector": vector, "limit": top_k, "with_payload": True},
        timeout=10,
    )
    resp.raise_for_status()
    results = []
    for hit in resp.json()["result"]:
        text = hit["payload"].get("text", "") or hit["payload"].get("summary", "")
        results.append(
            SearchResult(id=str(hit["id"]), text=text, score=hit["score"], metadata=hit["payload"])
        )
    return results


# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

def _rrf_fuse(
    *result_lists: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    best: dict[str, SearchResult] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (k + rank + 1)
            if r.id not in best:
                best[r.id] = r
    fused = []
    for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        entry = best[doc_id]
        fused.append(SearchResult(id=entry.id, text=entry.text, score=score, metadata=entry.metadata))
    return fused


# ---------------------------------------------------------------------------
# Cross-Encoder Reranker
# ---------------------------------------------------------------------------

def _rerank(query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
    if not results:
        return []
    pairs = [(query, r.text) for r in results]
    scores = reranker.predict(pairs)
    for i, r in enumerate(results):
        r.score = float(scores[i])
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# 통합 검색 인터페이스
# ---------------------------------------------------------------------------

def hybrid_search(query: str, top_k: int = 5) -> list[SearchResult]:
    """
    Hybrid Search 파이프라인:
    1. Elasticsearch BM25 키워드 검색 (nori 형태소, top-20)
    2. 벡터 유사도 검색 (top-20)
    3. RRF로 결과 융합
    4. Cross-Encoder Reranker로 최종 top-k 선정
    """
    logger.info("Hybrid search query=%s top_k=%d", query[:80], top_k)
    es_results = _es_search(query, top_k=20)
    vector_results = _vector_search(query, top_k=20)
    fused = _rrf_fuse(es_results, vector_results)
    reranked = _rerank(query, fused, top_k=top_k)
    logger.info("Hybrid search returned %d results (es=%d, vec=%d)", len(reranked), len(es_results), len(vector_results))
    return reranked
