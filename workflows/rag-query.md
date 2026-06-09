# RAG Q&A Flow

질문 → LangGraph Agentic RAG (쿼리 분석 → Hybrid Search → 생성 → 반성 루프) → 답변.

## Flow

1. **Trigger**: webhook으로 `{"question": "지난 회의에서 결정된 사항은?"}` 수신.

2. **RAG Query** (Agentic RAG 파이프라인):
   ```
   POST http://rag:8001/query
   {"question": "지난 회의에서 결정된 사항은?"}
   ```

   내부 동작:
   - **쿼리 분석**: LLM이 라우팅(검색/직접 답변) 결정 + 복합 쿼리 분해
   - **Hybrid Search**: BM25 키워드 검색 + 벡터 유사도 검색 → RRF 융합 → Cross-Encoder Reranker
   - **답변 생성**: 검색된 컨텍스트 기반으로 LLM이 답변 생성
   - **반성 루프**: 답변의 관련성·충실성 자체 평가, 불합격 시 쿼리 재구성 후 재검색 (최대 2회)

   응답:
   ```json
   {
     "answer": "지난 회의에서 결정된 사항은...",
     "contexts": [
       {"id": "uuid", "text": "관련 문맥", "score": 0.95}
     ]
   }
   ```

3. **(선택) 품질 평가**:
   ```
   POST http://eval:8002/evaluate
   {"sample": {"question": "결정 사항은?", "ground_truth": "예산 승인"}}
   ```
   - answer/contexts 미제공 시 RAG 서비스에서 자동 조회
   - Faithfulness, Answer Relevance, Context Precision, Context Recall 자동 측정

## LangGraph 파이프라인 구조

```
analyse_query → route
  ├─ (direct) → generate_direct → END
  └─ (retrieve) → retrieve → generate → reflect
                                           ├─ (accept) → END
                                           └─ (retry, max 2) → rewrite_query → retrieve → ...
```

## n8n 워크플로우 구성

- **Trigger Node**: Webhook (POST, JSON body)
- **HTTP Request Node (Query)**: `POST http://rag:8001/query` → answer + contexts 추출
- **Respond to Webhook**: 답변 반환
