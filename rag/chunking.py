"""Semantic Chunking module.

Splits documents based on embedding similarity between consecutive sentences
rather than fixed token/character windows, preserving semantic coherence.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import httpx
import numpy as np

from rag.config import EMBED_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger("rag.chunking")


@dataclass
class SemanticChunk:
    text: str
    metadata: dict = field(default_factory=dict)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (supports Korean punctuation)."""
    # Split on period, question mark, exclamation mark followed by space or end
    sentences = re.split(r"(?<=[.?!。？！])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Get embeddings from Ollama for a batch of texts."""
    embeddings = []
    with httpx.Client(timeout=120.0) as client:
        for text in texts:
            resp = client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
    return embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a)
    vb = np.array(b)
    dot = np.dot(va, vb)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def semantic_chunk(
    text: str,
    breakpoint_threshold: float = 0.5,
    metadata: dict | None = None,
) -> list[SemanticChunk]:
    """Split text into semantic chunks based on embedding similarity.

    Algorithm:
    1. Split text into sentences.
    2. Compute embeddings for each sentence.
    3. Calculate cosine similarity between consecutive sentence embeddings.
    4. Where similarity drops below the threshold, create a chunk boundary.
    5. Group sentences between boundaries into chunks.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        return [SemanticChunk(text=sentences[0], metadata=metadata or {})]

    logger.info("Computing embeddings for %d sentences", len(sentences))
    embeddings = _embed_texts(sentences)

    # Compute similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # Find breakpoints where similarity drops below threshold
    chunks: list[SemanticChunk] = []
    current_sentences: list[str] = [sentences[0]]

    for i, sim in enumerate(similarities):
        if sim < breakpoint_threshold:
            # Create chunk from accumulated sentences
            chunk_text = " ".join(current_sentences)
            chunk_meta = {**(metadata or {}), "chunk_index": len(chunks)}
            chunks.append(SemanticChunk(text=chunk_text, metadata=chunk_meta))
            current_sentences = [sentences[i + 1]]
        else:
            current_sentences.append(sentences[i + 1])

    # Don't forget the last chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunk_meta = {**(metadata or {}), "chunk_index": len(chunks)}
        chunks.append(SemanticChunk(text=chunk_text, metadata=chunk_meta))

    logger.info(
        "Created %d semantic chunks from %d sentences (threshold=%.2f)",
        len(chunks),
        len(sentences),
        breakpoint_threshold,
    )
    return chunks
