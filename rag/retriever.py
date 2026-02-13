"""Hybrid Retriever: BM25 + Vector Search + Cross-Encoder Reranker.

Combines sparse (BM25) and dense (vector) retrieval with a cross-encoder
reranker to maximise both recall and precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from rag.config import (
    COLLECTION_NAME,
    EMBED_MODEL,
    EMBEDDING_DIM,
    OLLAMA_BASE_URL,
    QDRANT_URL,
    RERANKER_MODEL,
    TOP_K_BM25,
    TOP_K_RERANK,
    TOP_K_VECTOR,
)

logger = logging.getLogger("rag.retriever")


@dataclass
class RetrievedDoc:
    id: str
    text: str
    score: float
    metadata: dict


# ---------------------------------------------------------------------------
# Clients (lazy-initialised at module level for container reuse)
# ---------------------------------------------------------------------------
_qdrant: QdrantClient | None = None
_reranker: CrossEncoder | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=QDRANT_URL, timeout=30)
    return _qdrant


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading reranker model: %s", RERANKER_MODEL)
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------
def embed_text(text: str) -> list[float]:
    """Get embedding from Ollama."""
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        resp.raise_for_status()
    return resp.json()["embedding"]


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------
def ensure_collection() -> None:
    """Create the Qdrant collection if it does not exist."""
    qdrant = _get_qdrant()
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM, distance=Distance.COSINE
            ),
        )
        logger.info("Created collection '%s'", COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def ingest_chunks(
    chunks: list[dict],
) -> int:
    """Store chunked documents into Qdrant.

    Each chunk dict must have: id, text, embedding, metadata.
    Returns the number of points upserted.
    """
    ensure_collection()
    qdrant = _get_qdrant()

    points = [
        PointStruct(
            id=chunk["id"],
            vector=chunk["embedding"],
            payload={"text": chunk["text"], **chunk.get("metadata", {})},
        )
        for chunk in chunks
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Upserted %d chunks into '%s'", len(points), COLLECTION_NAME)
    return len(points)


# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------
def vector_search(query_embedding: list[float], top_k: int = TOP_K_VECTOR) -> list[RetrievedDoc]:
    qdrant = _get_qdrant()
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
        with_payload=True,
    )
    docs = []
    for hit in results:
        payload = hit.payload or {}
        docs.append(
            RetrievedDoc(
                id=str(hit.id),
                text=payload.get("text", ""),
                score=hit.score,
                metadata={k: v for k, v in payload.items() if k != "text"},
            )
        )
    return docs


# ---------------------------------------------------------------------------
# BM25 search
# ---------------------------------------------------------------------------
def bm25_search(
    query: str, corpus_docs: list[RetrievedDoc], top_k: int = TOP_K_BM25
) -> list[RetrievedDoc]:
    """Run BM25 over a pre-fetched corpus (typically vector search results
    expanded to a larger candidate set)."""
    if not corpus_docs:
        return []

    tokenized_corpus = [doc.text.split() for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    scored = list(zip(corpus_docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    results = []
    for doc, score in scored[:top_k]:
        results.append(
            RetrievedDoc(
                id=doc.id,
                text=doc.text,
                score=float(score),
                metadata=doc.metadata,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Hybrid search + Reranker
# ---------------------------------------------------------------------------
def _reciprocal_rank_fusion(
    result_lists: list[list[RetrievedDoc]], k: int = 60
) -> list[RetrievedDoc]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF)."""
    scores: dict[str, float] = {}
    doc_map: dict[str, RetrievedDoc] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc.id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [
        RetrievedDoc(
            id=did,
            text=doc_map[did].text,
            score=scores[did],
            metadata=doc_map[did].metadata,
        )
        for did in sorted_ids
    ]


def hybrid_search(
    query: str,
    top_k: int = TOP_K_RERANK,
) -> list[RetrievedDoc]:
    """Execute hybrid search: Vector + BM25 → RRF fusion → Cross-Encoder rerank.

    Pipeline:
    1. Dense vector search (top-K candidates)
    2. BM25 sparse search over the same candidate set
    3. Reciprocal Rank Fusion to merge results
    4. Cross-encoder reranker for final ranking
    """
    # 1. Vector search
    query_embedding = embed_text(query)
    vector_results = vector_search(query_embedding, top_k=TOP_K_VECTOR)

    if not vector_results:
        return []

    # 2. BM25 over vector candidate set (we use a broader pool)
    # Fetch a larger pool for BM25 diversity
    large_pool = vector_search(query_embedding, top_k=max(TOP_K_VECTOR * 2, 20))
    bm25_results = bm25_search(query, large_pool, top_k=TOP_K_BM25)

    # 3. Reciprocal Rank Fusion
    fused = _reciprocal_rank_fusion([vector_results, bm25_results])

    # 4. Cross-encoder reranking
    reranker = _get_reranker()
    pairs = [(query, doc.text) for doc in fused]
    rerank_scores = reranker.predict(pairs)

    for doc, score in zip(fused, rerank_scores):
        doc.score = float(score)

    fused.sort(key=lambda d: d.score, reverse=True)

    final = fused[:top_k]
    logger.info(
        "Hybrid search: vector=%d, bm25=%d, fused=%d, reranked top=%d",
        len(vector_results),
        len(bm25_results),
        len(fused),
        len(final),
    )
    return final
