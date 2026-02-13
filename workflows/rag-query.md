# RAG Q&A (Agentic RAG Pipeline)

LangGraph-based agentic retrieval with hybrid search and self-reflection.

1) **Query**: `POST http://rag:8001/query` with `{ "question": "..." }`.
2) **Query Analysis** (LangGraph node):
   - Route decision: `retrieve` (knowledge-dependent) or `direct` (simple/general).
   - Query decomposition: complex questions split into 1-3 sub-queries.
3) **Hybrid Retrieval** (if routed to `retrieve`):
   - Dense vector search via Qdrant (top-K candidates).
   - BM25 sparse search over the candidate pool.
   - Reciprocal Rank Fusion (RRF) to merge ranked lists.
   - Cross-encoder reranker for final precision ranking.
4) **Generation**: Ollama `llama3` generates an answer grounded in retrieved context.
5) **Reflection Loop**:
   - Self-evaluates answer relevance and faithfulness.
   - If quality is insufficient, reformulates the query and retries (up to 2 retries).
6) **Response**: Returns answer, source documents, route taken, and retry count.

## Evaluation (optional)

After generating answers, quality can be measured via the eval service:

```bash
POST http://eval:8002/evaluate
{
  "question": "...",
  "answer": "...",
  "contexts": ["..."],
  "ground_truth": "..." (optional)
}
```

Metrics: Faithfulness, Answer Relevance, Context Precision, Context Recall (RAGAS).
