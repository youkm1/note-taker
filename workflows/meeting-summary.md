# Meeting Ingestion (Enhanced Pipeline)

End-to-end flow: audio → STT → semantic chunking → embedding → vector store.

1) **Trigger**: webhook or manual trigger receives the audio file (field `audio`) and optional `lang`.
2) **STT**: `POST http://stt:8000/stt` (multipart form with `audio`, `lang=ko`). Uses Whisper Large V3 Turbo (KsponSpeech fine-tuned) with VAD filtering.
3) **Ingest via RAG service**: `POST http://rag:8001/ingest` with `{text, title, source, metadata}`.
   - Semantic Chunking: splits transcript by embedding similarity (not fixed windows).
   - Auto-embeds each chunk via Ollama `nomic-embed-text`.
   - Stores in Qdrant with metadata (title, timestamp, source, chunk_index).
4) **(Optional) Quality check**: `POST http://eval:8002/evaluate` to verify ingestion quality.
