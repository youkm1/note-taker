"""Semantic Chunking — 임베딩 유사도 기반으로 텍스트를 의미 단위 청크로 분할."""

import logging
import re
from dataclasses import dataclass

import numpy as np
import requests

logger = logging.getLogger("rag.chunking")

OLLAMA_URL = "http://ollama:11434"
EMBED_MODEL = "nomic-embed-text"


@dataclass
class Chunk:
    text: str
    index: int


def _split_sentences(text: str) -> list[str]:
    """문장 단위로 분할 (한국어/영어 혼합 지원)."""
    parts = re.split(r"(?<=[.!?。\n])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _embed(texts: list[str]) -> np.ndarray:
    """Ollama nomic-embed-text를 이용해 텍스트 리스트의 임베딩 벡터를 반환."""
    vectors = []
    for t in texts:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=30,
        )
        resp.raise_for_status()
        vectors.append(resp.json()["embedding"])
    return np.array(vectors)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def semantic_chunk(
    text: str,
    similarity_threshold: float = 0.5,
    max_chunk_sentences: int = 10,
) -> list[Chunk]:
    """
    의미적 유사도 기반 청킹.

    연속된 문장 간 코사인 유사도가 threshold 이상이면 같은 청크로 합치고,
    아래로 떨어지면 새 청크를 시작합니다.
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [Chunk(text=text.strip(), index=0)] if text.strip() else []

    logger.info("Embedding %d sentences for semantic chunking", len(sentences))
    embeddings = _embed(sentences)

    chunks: list[Chunk] = []
    current_sentences: list[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim >= similarity_threshold and len(current_sentences) < max_chunk_sentences:
            current_sentences.append(sentences[i])
        else:
            chunks.append(Chunk(text=" ".join(current_sentences), index=len(chunks)))
            current_sentences = [sentences[i]]

    if current_sentences:
        chunks.append(Chunk(text=" ".join(current_sentences), index=len(chunks)))

    logger.info("Created %d semantic chunks from %d sentences", len(chunks), len(sentences))
    return chunks
