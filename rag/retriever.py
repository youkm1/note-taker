"""Hybrid Search (BM25 + 벡터) → RRF 융합 → Cross-Encoder Reranker."""

import logging
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
import requests
from sentence_transformers import CrossEncoder

logger = logging.getLogger("rag.retriever")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION = os.getenv("QDRANT_COLLECTION", "notes")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = CrossEncoder(RERANKER_MODEL)


@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BM25 (인메모리 간이 구현 — Qdrant 전체 문서 대상)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_dl: float,
    doc_count: int,
    df: dict[str, int],
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    dl = len(doc_tokens)
    score = 0.0
    tf_map: dict[str, int] = {}
    for t in doc_tokens:
        tf_map[t] = tf_map.get(t, 0) + 1
    for qt in query_tokens:
        if qt not in tf_map:
            continue
        tf = tf_map[qt]
        n = df.get(qt, 0)
        idf = math.log((doc_count - n + 0.5) / (n + 0.5) + 1.0)
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        score += idf * tf_norm
    return score


def _fetch_all_points() -> list[dict]:
    """Qdrant 컬렉션에서 전체 포인트를 스크롤로 가져옴."""
    points: list[dict] = []
    offset = None
    while True:
        body: dict = {"limit": 100, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        resp = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            json=body,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()["result"]
        points.extend(data["points"])
        offset = data.get("next_page_offset")
        if offset is None:
            break
    return points


def _bm25_search(query: str, top_k: int = 20) -> list[SearchResult]:
    points = _fetch_all_points()
    if not points:
        return []

    query_tokens = _tokenize(query)
    docs_tokens = []
    for p in points:
        text = p["payload"].get("text", "") or p["payload"].get("summary", "")
        docs_tokens.append(_tokenize(text))

    avg_dl = sum(len(d) for d in docs_tokens) / max(len(docs_tokens), 1)
    df: dict[str, int] = {}
    for dt in docs_tokens:
        seen: set[str] = set()
        for t in dt:
            if t not in seen:
                df[t] = df.get(t, 0) + 1
                seen.add(t)

    scored = []
    for i, p in enumerate(points):
        s = _bm25_score(query_tokens, docs_tokens[i], avg_dl, len(points), df)
        text = p["payload"].get("text", "") or p["payload"].get("summary", "")
        scored.append(SearchResult(id=str(p["id"]), text=text, score=s, metadata=p["payload"]))
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# 벡터 검색
# ---------------------------------------------------------------------------

def _embed_query(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _vector_search(query: str, top_k: int = 20) -> list[SearchResult]:
    vector = _embed_query(query)
    resp = requests.post(
        f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
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
    1. BM25 키워드 검색 (top-20)
    2. 벡터 유사도 검색 (top-20)
    3. RRF로 결과 융합
    4. Cross-Encoder Reranker로 최종 top-k 선정
    """
    logger.info("Hybrid search query=%s top_k=%d", query[:80], top_k)
    bm25_results = _bm25_search(query, top_k=20)
    vector_results = _vector_search(query, top_k=20)
    fused = _rrf_fuse(bm25_results, vector_results)
    reranked = _rerank(query, fused, top_k=top_k)
    logger.info("Hybrid search returned %d results", len(reranked))
    return reranked
