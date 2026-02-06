# RAG Q&A (n8n Example)

Conceptual flow for answering questions against stored meeting notes.

1) Trigger: webhook receives `{ "question": "..." }`.
2) Embed question: HTTP Request node → `POST http://ollama:11434/api/embeddings` with the question using `nomic-embed-text`.
3) Vector search: HTTP Request node → Qdrant `POST http://qdrant:6333/collections/notes/points/search` with the embedding to fetch top-k similar notes.
4) Build prompt: Merge retrieved notes (transcript + summary snippets) into a context string plus the user question, instruct the model to answer only from that context.
5) Answer: HTTP Request node → `POST http://ollama:11434/api/chat` with the context-enriched prompt using `llama3`, return the generated answer as the webhook response.
