# Meeting Ingestion (n8n Example)

Conceptual flow for turning an uploaded meeting audio file into a summarized, searchable note.

1) Trigger: webhook or manual trigger receives the audio file (field `audio`) and optional `lang`.
2) STT: HTTP Request node → `POST http://stt:8000/stt` (multipart form with `audio`, query `lang` if provided) → returns transcript text.
3) Summarize: HTTP Request node → `POST http://ollama:11434/api/chat` with a prompt that summarizes the transcript using the `llama3` model.
4) Embed: HTTP Request node → `POST http://ollama:11434/api/embeddings` with the transcript or summary using the `nomic-embed-text` model → returns vector.
5) Persist: HTTP Request node → Qdrant `PUT`/`POST` to store `{id, timestamp, title, transcript, summary, embedding}` in a collection (e.g., `notes`).
