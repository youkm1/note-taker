## 개요

기존 Whisper small 기반 STT + n8n 오케스트레이션 구조에서, **End-to-End RAG 파이프라인**으로 전면 업그레이드합니다.

| 영역 | Before | After |
|------|--------|-------|
| STT 모델 | Whisper `small`, beam_size=2 | Whisper `large-v3-turbo`, beam_size=5, VAD 필터 |
| 한국어 지원 | 없음 | KsponSpeech 파인튜닝 모델 지원 (`WHISPER_HF_MODEL`) |
| 청킹 | 없음 (n8n에서 전체 텍스트 처리) | Semantic Chunking (임베딩 유사도 기반 의미 단위 분할) |
| 검색 | 단순 벡터 검색 | Hybrid Search (BM25 + 벡터) → RRF 융합 → Cross-Encoder Reranker |
| 답변 생성 | 단순 프롬프트 | LangGraph Agentic RAG (쿼리 분해, 조건 분기, 반성 루프 최대 2회) |
| 품질 평가 | 수동 | RAGAS 자동 측정 (Faithfulness, Answer Relevance, Context Precision/Recall) |
| LLM/임베딩 | Ollama (llama3, nomic-embed-text, 768차원) | Gemini API (gemini-2.5-flash, gemini-embedding-001, 3072차원) |
| 서비스 수 | 4개 (STT, Qdrant, Ollama, n8n) | 5개 (STT, Qdrant, RAG, Eval, n8n) + Gemini API |

### 변경 파일 (18 files, +1272, -110)

**수정**: `stt/app.py`, `stt/requirements.txt`, `docker-compose.yml`, `ARCHITECTURE.md`, `README.md`, `workflows/*.md`
**신규**: `rag/` (chunking.py, retriever.py, graph.py, app.py, Dockerfile, requirements.txt), `eval/` (app.py, Dockerfile, requirements.txt)

---

## TODO

- [x] STT 서비스 업그레이드 (large-v3-turbo, KsponSpeech, VAD, /health)
- [x] RAG 서비스 구축 — Semantic Chunking (`rag/chunking.py`)
- [x] RAG 서비스 구축 — Hybrid Search + RRF + Reranker (`rag/retriever.py`)
- [x] LangGraph Agentic RAG 구현 (`rag/graph.py`)
- [x] RAG FastAPI 엔드포인트 (`rag/app.py` — POST /ingest, /query)
- [x] RAGAS 평가 서비스 구축 (`eval/app.py` — POST /evaluate, /evaluate/batch)
- [x] Docker Compose에 rag, eval 서비스 추가 + 의존성 설정
- [x] 문서 전면 업데이트 (ARCHITECTURE.md, README.md, workflows/)

---

## 개선한 이유

### 1. STT: `small` → `large-v3-turbo` + KsponSpeech
- **문제**: small 모델은 한국어 인식률이 낮고, 무음 구간까지 처리하여 불필요한 지연 발생.
- **해결**: large-v3-turbo는 large-v3 수준 정확도에 **4배 빠른 추론**. VAD 필터로 무음 자동 스킵. `WHISPER_HF_MODEL` 환경변수로 KsponSpeech 파인튜닝 모델을 플러그인 방식으로 교체 가능.
- **beam_size 2→5**: 탐색 후보를 넓혀 전사 정확도를 높임.

### 2. Semantic Chunking (고정 윈도우 → 의미 단위 분할)
- **문제**: 고정 크기 청킹은 문장 중간에서 잘려 의미가 훼손되고, 검색 시 관련 없는 컨텍스트가 섞임.
- **해결**: 연속 문장 간 임베딩 코사인 유사도를 측정해, threshold(0.5) 이상이면 같은 청크로 합침. 의미적 일관성이 유지되어 검색 precision 향상.

### 3. Hybrid Search + Reranker (단순 벡터 → BM25+벡터+RRF+Reranker)
- **문제**: 벡터 검색만으로는 키워드 정확 매칭을 놓치고, 임베딩 유사도만으로 순위를 매기면 false positive 발생.
- **해결**:
  - BM25 (키워드 TF-IDF) + 벡터 (의미 유사도) 각각 top-20 검색
  - RRF(Reciprocal Rank Fusion)로 두 결과의 순위를 융합 → 단일 방법 편향 방지
  - Cross-Encoder(`ms-marco-MiniLM-L-6-v2`)로 최종 재순위 → bi-encoder보다 정밀한 관련성 판단

### 4. LangGraph Agentic RAG (단순 생성 → 반성 루프)
- **문제**: 한 번의 검색+생성으로 hallucination이나 관련 없는 답변이 나올 수 있음.
- **해결**:
  - **쿼리 분석**: 검색이 필요한지 라우팅 + 복합 질문을 1-3개 서브쿼리로 분해
  - **반성 루프**: 생성된 답변의 관련성·충실성을 LLM이 자체 평가, 불합격 시 쿼리 재구성 후 재검색 (최대 2회)
  - 그래프 구조: `analyse_query → route → retrieve → generate → reflect → (retry or END)`

### 5. RAGAS 평가 파이프라인
- **문제**: RAG 품질을 정량적으로 측정할 수단이 없어, 개선 효과를 검증할 수 없음.
- **해결**: Faithfulness, Answer Relevance, Context Precision, Context Recall 4개 메트릭 자동 측정. 배치 평가로 전체 시스템 벤치마킹 가능.

---

## 트러블슈팅

### STT 모델 로딩 실패
```
whisper model not found
```
→ `WHISPER_HF_MODEL`에 CTranslate2 변환 모델 경로를 지정해야 함. 일반 HuggingFace 모델은 faster-whisper와 호환 안 됨. ct2 변환 모델 사용 필요 (예: `seastar105/whisper-large-v3-turbo-ksponspeech-ct2`).

### Qdrant 컬렉션 벡터 차원 불일치
```
Wrong input: Vector dimension error
```
→ `VECTOR_SIZE` 환경변수(기본 3072)가 실제 임베딩 모델 출력 차원과 일치해야 함. Gemini embedding-001은 3072차원.

### Cross-Encoder 모델 다운로드 지연
```
OSError: Can't load tokenizer for 'cross-encoder/ms-marco-MiniLM-L-6-v2'
```
→ 최초 실행 시 HuggingFace에서 모델 다운로드 필요. 오프라인 환경에서는 사전 다운로드 후 로컬 경로를 `RERANKER_MODEL`에 지정.

### LangGraph 무한 루프 방지
→ `MAX_RETRIES = 2`로 반성 루프 재시도 횟수를 제한. 2회 초과 시 현재 답변을 그대로 반환.

### Docker Compose 서비스 기동 순서
→ `rag`는 `qdrant`에 의존하고 Gemini API를 사용. `eval`은 `rag`에 의존. `depends_on`으로 순서를 보장하지만, 실제 서비스 ready 상태는 보장하지 않으므로 health check 엔드포인트로 확인 후 요청.
