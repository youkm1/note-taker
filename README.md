# Local AI Note Taker (v2)

Self-hosted Korean meeting transcription & RAG system with agentic retrieval and automated quality evaluation.

## Architecture Overview

- **STT**: Whisper Large V3 Turbo (KsponSpeech fine-tuned) via FastAPI
- **RAG Pipeline**: LangGraph-based Agentic RAG with Semantic Chunking, Hybrid Search (BM25 + Vector), and Cross-Encoder Reranker
- **LLM & Embeddings**: Ollama (`llama3` + `nomic-embed-text`)
- **Vector Store**: Qdrant
- **Evaluation**: RAGAS (Faithfulness, Answer Relevance, Context Precision, Context Recall)
- **Orchestration**: n8n

## Requirements

- Docker and docker-compose installed locally.

## Run locally

1. `cp .env.example .env` (adjust values if desired).
2. `docker-compose up -d`
3. Open `http://localhost:5678` for the n8n UI.

## Services

| Service | Port | Endpoint |
|---------|------|----------|
| STT | 8000 | `POST /stt` — Korean speech-to-text |
| RAG | 8001 | `POST /ingest` — document ingestion, `POST /query` — agentic RAG query |
| Eval | 8002 | `POST /evaluate` — RAGAS single eval, `POST /evaluate/batch` — batch eval |
| Ollama | 11434 | `POST /api/chat`, `POST /api/embeddings` |
| Qdrant | 6333 | Vector DB HTTP API |
| n8n | 5678 | Workflow orchestration UI |

## Quick Start

```bash
# Transcribe Korean audio
curl -X POST http://localhost:8000/stt -F "audio=@meeting.wav" -F "lang=ko"

# Ingest a document
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "회의 내용...", "title": "전략 회의"}'

# Query with Agentic RAG
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "전략 회의에서 결정된 사항은?"}'

# Evaluate RAG quality
curl -X POST http://localhost:8002/evaluate \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "answer": "...", "contexts": ["..."]}'
```

## Key Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | Whisper model size |
| `WHISPER_HF_MODEL` | (empty) | HuggingFace fine-tuned model ID for Korean |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `LLM_MODEL` | `llama3` | Ollama chat model |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |

## Example Workflows

- **Meeting ingestion**: `workflows/meeting-summary.md`
- **RAG Q&A**: `workflows/rag-query.md`

## Data Persistence

- Qdrant data: `qdrant_data/`
- Ollama models: `ollama_data/`
- n8n state: `n8n_data/`

See `ARCHITECTURE.md` for detailed Korean documentation.
