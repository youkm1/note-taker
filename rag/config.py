import os


# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ---------------------------------------------------------------------------
# Qdrant collection
# ---------------------------------------------------------------------------
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "notes")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))  # nomic-embed-text

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "10"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "10"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_BREAKPOINT_THRESHOLD = float(
    os.getenv("CHUNK_BREAKPOINT_THRESHOLD", "0.5")
)
