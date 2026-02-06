# Local AI Note Taker

This stack mirrors a self-hosted "CLOVA Note": upload an audio file, transcribe it locally, summarize it with a local LLM, embed the content, and store it for retrieval-augmented Q&A.

## What it does
- Audio ingestion: FastAPI + Faster-Whisper transcribes uploads at `http://localhost:8000/stt`.
- Summarization + embeddings: Ollama serves `llama3` for chat/summaries and `nomic-embed-text` for vectors at `http://localhost:11434`.
- Storage: Qdrant keeps transcripts, summaries, and embeddings for similarity search.
- Orchestration: n8n wires the calls together (UI at `http://localhost:5678`).
- RAG Q&A: A question is embedded via Ollama, matched in Qdrant, and answered by Ollama using only retrieved context.

## Requirements
- Docker and docker-compose installed locally.

## Run locally
1. `cp .env.example .env` (adjust values if desired).
2. `docker-compose up -d`.
3. Open `http://localhost:5678` for the n8n UI (basic auth uses values from `.env`).

## Services
- STT: `POST http://localhost:8000/stt` (multipart `audio`, optional `lang`).
- Ollama: chat at `POST /api/chat`, embeddings at `POST /api/embeddings` on `http://localhost:11434`.
- Qdrant: HTTP API on `http://localhost:6333`.
- n8n: UI on `http://localhost:5678`.

## Example workflows
- Meeting ingestion: see `workflows/meeting-summary.md` (Trigger → STT → summarize → embed → store in Qdrant).
- RAG Q&A: see `workflows/rag-query.md` (Question → embed → Qdrant search → prompt build → Ollama answer).

## Data persistence
- Qdrant data: `qdrant_data/`
- Ollama models: `ollama_data/`
- n8n state: `n8n_data/`
