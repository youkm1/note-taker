# Local AI Note Taker 아키텍처 정리

도커 컴포즈로 STT(Whisper), RAG(Semantic Chunking + Hybrid Search + LangGraph), 벡터 DB(Qdrant), 평가(RAGAS), 오케스트레이션(n8n) 서비스를 구성하고, LLM/임베딩은 Gemini API를 활용하는 End-to-End RAG 노트 테이커입니다. STT는 Faster-Whisper로 완전 로컬 처리하여 민감한 음성 데이터의 외부 유출을 방지합니다.

> **설계 변천**: 초기(V1)에는 Ollama(llama3 + nomic-embed-text)로 완전 로컬 구성이었으나, 한국어 생성 품질과 임베딩 정확도 향상을 위해 Gemini API로 전환. STT는 원시 음성 데이터 보호를 위해 로컬을 유지.

## 전체 구조

```
┌─────────────────────────────────────────────────────────┐
│                  Docker Compose Network                   │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │   STT    │  │   RAG    │  │   Eval   │  │   n8n   │ │
│  │  :8000   │  │  :8001   │  │  :8002   │  │  :5678  │ │
│  │ Whisper  │  │ LangGraph│  │  RAGAS   │  │Workflow │ │
│  │ large-v3 │  │ Hybrid   │  │          │  │         │ │
│  │ -turbo   │  │ Search   │  │          │  │         │ │
│  └──────────┘  └────┬─────┘  └────┬─────┘  └─────────┘ │
│                     │             │                      │
│              ┌──────┴─────────────┘                      │
│              │                                           │
│         ┌────┴─────┐                                     │
│         │  Qdrant  │                                     │
│         │  :6333   │                                     │
│         │  Vector  │                                     │
│         │  Store   │                                     │
│         └──────────┘                                     │
└─────────────────────────────────────────────────────────┘
          │                          │
          └──── Gemini API ──────────┘
               (LLM: gemini-2.5-flash)
               (Embedding: gemini-embedding-001)
```

- **네트워크**: 단일 도커 브리지 네트워크에서 서비스끼리 컨테이너 이름으로 통신.
- **포트 매핑**: STT 8000, RAG 8001, Eval 8002, Qdrant 6333, n8n 5678.
- **스토리지**: `qdrant_data/`, `n8n_data/`로 영속 볼륨 마운트.
- **외부 의존**: Gemini API (LLM 생성 + 임베딩).

## 기술 스택과 채택 이유

### STT: FastAPI + faster-whisper (large-v3-turbo)
- **이유**: Faster-Whisper가 공식 구현 대비 속도/메모리 효율이 높음. large-v3-turbo는 large-v3 수준의 정확도에 4배 빠른 추론 속도.
- **KsponSpeech 지원**: `WHISPER_HF_MODEL` 환경변수로 공개된 한국어 특화 fine-tuned 모델(예: `seastar105/whisper-large-v3-turbo-ksponspeech-ct2`) 적용 가능. 설정 시 한국어 자동 기본 언어.
- **int8 추론**: `compute_type=int8` 옵션으로 CPU 환경에서 메모리/속도 최적화.
- **VAD 필터**: 무음 구간을 자동 스킵하여 처리 속도 향상.
- **병렬 청킹**: 긴 오디오를 30초 단위(2초 overlap)로 분할 후 병렬 전사.
- **사용**: `POST /stt` (multipart `audio`, 옵션 `lang`), `GET /health`.

### RAG: Semantic Chunking + Hybrid Search + LangGraph Agentic RAG
- **Semantic Chunking** (`rag/chunking.py`): 고정 윈도우 대신 Gemini 임베딩 유사도 기반으로 의미적 일관성을 유지하면서 텍스트 분할. 연속 문장 간 코사인 유사도가 threshold(0.5) 이상이면 같은 청크로 합침.
- **Hybrid Search** (`rag/retriever.py`):
  - BM25 키워드 검색 + 벡터 유사도 검색을 각각 top-20으로 실행
  - RRF(Reciprocal Rank Fusion, k=60)로 두 결과를 융합
  - Cross-Encoder Reranker(`ms-marco-MiniLM-L-6-v2`)로 최종 top-k 선정
- **LangGraph Agentic RAG** (`rag/graph.py`):
  - 쿼리 분석: 라우팅(검색 vs 직접 답변) + 복합 쿼리 분해(1-3개 서브쿼리)
  - 조건 분기: 검색 필요 시 retrieve → generate, 불필요 시 직접 답변
  - 반성 루프: 생성된 답변의 관련성/충실성 자체 평가, 불합격 시 쿼리 재구성 후 재검색(최대 2회)
- **사용**: `POST /ingest` (텍스트 저장), `POST /query` (질의).

### RAGAS 평가 서비스
- **메트릭**: Faithfulness, Answer Relevance, Context Precision, Context Recall 자동 측정.
- **RAG 연동**: answer/contexts가 없으면 RAG 서비스에서 자동 조회.
- **평가 LLM/임베딩**: Gemini 2.5 Flash + Gemini Embedding 001 사용.
- **사용**: `POST /evaluate` (단일), `POST /evaluate/batch` (배치).

### LLM & 임베딩: Gemini API
- **LLM**: `gemini-2.5-flash` — RAG 답변 생성, 쿼리 분석, 반성 평가에 사용.
- **임베딩**: `gemini-embedding-001` (3072차원) — Semantic Chunking, 벡터 검색, RAGAS 평가에 사용.
- **변천 이유**: 초기에는 Ollama(llama3 + nomic-embed-text)로 완전 로컬 구성이었으나, llama3의 한국어 품질 한계와 nomic-embed-text(768차원) 대비 Gemini embedding의 검색 정확도 우위로 전환. 프로덕션에서는 on-premise LLM(예: IBM Granite)으로 대체 가능한 구조.

### 벡터 스토어: Qdrant
- **이유**: 가벼운 컨테이너 배포와 빠른 k-NN/스코어링, REST/gRPC API.
- **사용**: 컬렉션 자동 생성(RAG 서비스 startup), `points/search`로 검색.

### 오케스트레이션/UI: n8n
- **이유**: 시각적 워크플로우 편집기로 HTTP 호출과 데이터 가공을 노코드로 연결.
- **사용**: `http://localhost:5678`.

### 인프라: Docker Compose
- **이유**: 단일 명령(`docker-compose up -d`)으로 5개 서비스를 일관 배포.

## 동작 흐름

### 업로드 → 저장 (`workflows/meeting-summary.md`)
1. 업로드 트리거(webhook/수동)에서 오디오 수신.
2. STT 서비스(Whisper large-v3-turbo)로 전송 → VAD 필터 + 병렬 청킹으로 전사 텍스트 반환.
3. RAG 서비스 `/ingest`로 전사 텍스트 전송 → Semantic Chunking → Gemini 임베딩 → Qdrant 저장.

### 질의 → Agentic RAG → 응답 (`workflows/rag-query.md`)
1. 질문 수신 → RAG 서비스 `/query` 호출.
2. **쿼리 분석**: Gemini LLM이 라우팅(검색/직접) 결정 + 복합 쿼리 분해.
3. **검색**: 서브쿼리별 Hybrid Search(BM25 + 벡터) → RRF 융합 → Reranker.
4. **생성**: 검색 결과를 컨텍스트로 Gemini LLM이 답변 생성.
5. **반성**: 답변의 관련성·충실성을 자체 평가. 불합격 시 쿼리 재구성 후 재검색(최대 2회 반복).
6. 최종 답변 반환.

### 품질 평가 (RAGAS)
1. `POST /evaluate` 또는 `/evaluate/batch`로 QA 쌍 제출.
2. answer/contexts 미제공 시 RAG 서비스에서 자동 조회.
3. Faithfulness, Answer Relevance, Context Precision, Context Recall 자동 측정.

## 주요 설정 포인트

| 환경변수 | 기본값 | 설명 |
|---------|--------|------|
| `WHISPER_MODEL_SIZE` | `large-v3-turbo` | Whisper 모델 크기 |
| `WHISPER_HF_MODEL` | (비어있음) | HuggingFace fine-tuned 모델 경로 (설정 시 MODEL_SIZE 무시) |
| `WHISPER_DEVICE` | `cpu` | `cpu` / `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | 정밀도 (`int8`, `float16`, `float32`) |
| `GEMINI_API_KEY` | (필수) | Gemini API 키 (RAG 서비스용) |
| `GEMINI_API_KEY_EVAL` | (필수) | Gemini API 키 (평가 서비스용, 별도 할당량) |
| `QDRANT_COLLECTION` | `notes` | Qdrant 컬렉션명 |
| `VECTOR_SIZE` | `3072` | 임베딩 벡터 차원 (Gemini embedding-001) |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-Encoder Reranker 모델 |

## 의사결정 기록

### 오디오 파일은 binary, 텍스트/메타데이터는 JSON으로 전달

- **결정**: m4a/wav 같은 오디오 파일은 `multipart/form-data`의 binary file로 전달하고, STT 이후의 전사 텍스트·질문·title·metadata·평가 요청은 JSON으로 전달한다.
- **이유**: 오디오 파일을 JSON에 base64로 넣으면 용량이 증가하고 인코딩/디코딩 비용과 메모리 사용량이 커진다. 반대로 전사 텍스트와 metadata는 구조화된 값이므로 JSON이 읽기 쉽고 API 계약을 명확하게 유지할 수 있다.
- **현재 구현**:
  - `POST /stt`: `audio: UploadFile = File(...)`로 파일을 받으므로 `multipart/form-data`를 사용한다.
  - `POST /ingest`: STT가 반환한 `{ "text": "..." }`를 기반으로 `{ "text", "title", "metadata" }` JSON을 RAG 서비스에 보낸다.
  - `POST /query`, `POST /evaluate`: 질문과 평가 입력은 JSON으로 보낸다.
- **n8n 적용**: Webhook 또는 Form Trigger에서 받은 오디오 파일은 n8n의 binary data로 유지하고, HTTP Request 노드에서 `formBinaryData`로 `stt:8000/stt`에 전달한다. 사용자 업로드 UI가 필요하면 Webhook 테스트 요청 대신 Form Trigger의 file field를 사용한다.
- **운영 메모**: 큰 오디오 파일을 다룰 때 n8n이 binary를 메모리에 오래 들고 있지 않도록 `N8N_DEFAULT_BINARY_DATA_MODE=filesystem` 설정을 권장한다.

## 인터페이스 예시

- STT:
  ```bash
  curl -X POST "http://localhost:8000/stt" \
       -F "audio=@sample.wav" -F "lang=ko"
  ```
- STT Health:
  ```bash
  curl http://localhost:8000/health
  ```
- RAG Ingest:
  ```bash
  curl -X POST http://localhost:8001/ingest \
       -H "Content-Type: application/json" \
       -d '{"text": "회의 전사 내용...", "title": "주간 회의"}'
  ```
- RAG Query:
  ```bash
  curl -X POST http://localhost:8001/query \
       -H "Content-Type: application/json" \
       -d '{"question": "지난 회의에서 결정된 사항은?"}'
  ```
- RAGAS 평가:
  ```bash
  curl -X POST http://localhost:8002/evaluate \
       -H "Content-Type: application/json" \
       -d '{"sample": {"question": "결정 사항은?", "ground_truth": "예산 승인"}}'
  ```
- RAGAS 배치 평가:
  ```bash
  curl -X POST http://localhost:8002/evaluate/batch \
       -H "Content-Type: application/json" \
       -d '{"samples": [{"question": "Q1", "ground_truth": "A1"}, {"question": "Q2", "ground_truth": "A2"}]}'
  ```

## 운영 지표·트러블슈팅 메모

- **로깅**: 모든 서비스는 `docker-compose logs -f <service>`로 모니터링.
- **헬스체크**: `curl http://localhost:{8000,8001,8002}/health`로 각 서비스 상태 확인.
- **STT 지연**: CPU에서 large-v3-turbo는 small보다 느릴 수 있음 → GPU 사용 또는 `WHISPER_MODEL_SIZE=small`로 변경.
- **Reranker 초기화**: 최초 실행 시 Cross-Encoder 모델 다운로드 필요.
- **CTranslate2 모델 필수**: `WHISPER_HF_MODEL` 사용 시 일반 HuggingFace 모델은 faster-whisper와 호환 안 됨. `-ct2` suffix가 붙은 CTranslate2 변환 모델 필요.

## 이력서에 적기 좋은 한줄 포인트

- Whisper Large V3 Turbo + KsponSpeech fine-tuned 모델 적용 및 FastAPI STT, Qdrant를 Docker Compose로 구성한 End-to-End RAG 파이프라인 설계.
- Semantic Chunking 및 Hybrid Search(BM25 + 벡터) + Cross-Encoder Reranker 적용으로 RAG 검색 품질 고도화.
- LangGraph 기반 Agentic RAG 구현 (쿼리 분해, 조건 분기 재검색, 반성 루프).
- RAGAS 기반 RAG 품질 평가 파이프라인 구축 (Faithfulness, Answer Relevance 메트릭 자동 측정).
