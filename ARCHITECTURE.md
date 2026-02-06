# Local AI Note Taker 아키텍처 정리

도커 컴포즈로 STT(Whisper), LLM/임베딩(Ollama), 벡터 DB(Qdrant), 오케스트레이션(n8n) 네 가지 서비스를 로컬에서 묶어 self-hosted 메모/요약/RAG 환경을 제공합니다.

## 전체 구조
- 네트워크: 단일 도커 브리지 네트워크에서 서비스끼리 컨테이너 이름으로 통신 (`stt`, `ollama`, `qdrant`, `n8n`).
- 포트 매핑: STT 8000, Ollama 11434, Qdrant 6333, n8n 5678 을 호스트에 노출.
- 스토리지: `qdrant_data/`, `ollama_data/`, `n8n_data/`로 영속 볼륨 마운트.
- 환경파일: `.env`에서 n8n Basic Auth 등을 주입, Whisper 관련 값은 docker-compose 기본값(모델 `small`, CPU, `int8`).

## 기술 스택과 채택 이유/사용법
- STT: FastAPI + `faster-whisper`  
  - 이유: 라우팅/에러 핸들링이 단순하며 Faster-Whisper가 공식 구현 대비 속도/메모리 효율이 높음.  
  - 사용: `POST /stt` (multipart `audio`, 옵션 `lang`). 환경변수로 모델 크기(`WHISPER_MODEL_SIZE`), 디바이스(`cpu`/`cuda`), 정밀도(`WHISPER_COMPUTE_TYPE`, 기본 `int8`) 조정.
- LLM & 임베딩: Ollama(`llama3`, `nomic-embed-text` 사전 pull)  
  - 이유: 인터넷 없이 로컬 모델 제공, REST API 호환(`api/chat`, `api/embeddings`). entrypoint에서 서버 기동 후 두 모델을 미리 당겨 cold-start를 줄임.  
  - 사용: Chat `POST /api/chat`, Embedding `POST /api/embeddings` (모델명 명시).
- 벡터 스토어: Qdrant  
  - 이유: 가벼운 컨테이너 배포와 빠른 k-NN/스코어링, REST/gRPC API.  
  - 사용: 컬렉션을 생성해 메타데이터(제목/요약/원문)와 벡터를 함께 저장 후 `points/search`로 검색.
- 오케스트레이션/UI: n8n  
  - 이유: 시각적 워크플로우 편집기로 HTTP 호출과 데이터 가공을 노코드로 연결, RAG 시나리오 반복 실험에 적합.  
  - 사용: 웹 UI(`http://localhost:5678`), Basic Auth는 `.env`로 설정.
- 인프라: Docker Compose  
  - 이유: 단일 명령(`docker-compose up -d`)으로 4개 서비스를 일관 배포, 각 컨테이너 자원/볼륨을 분리 관리.

## 동작 흐름
- 업로드→요약→저장 (`workflows/meeting-summary.md`)  
 1) 업로드 트리거(webhook/수동)에서 오디오 수신.  
 2) STT 서비스로 전송 → Whisper가 전사 텍스트 반환.  
 3) Ollama `llama3`로 요약/주요 포인트 생성.  
 4) `nomic-embed-text`로 임베딩 생성.  
 5) Qdrant 컬렉션에 `{id, timestamp, 제목, transcript, summary, embedding}` 저장.
- 질의→검색→응답 (`workflows/rag-query.md`)  
 1) 질문 수신.  
 2) 질문 임베딩 생성.  
 3) Qdrant에서 top-k 유사 노트 검색.  
 4) 검색 결과를 컨텍스트로 묶어 프롬프트 구성.  
 5) Ollama `llama3`가 컨텍스트 기반으로 답변 생성(출처 밖 정보 사용 금지).

## 주요 설정 포인트
- Whisper:
  - `WHISPER_MODEL_SIZE`로 정확도/속도 균형 조절 (`small`→빠름, `medium`/`large`→정확도 ↑, 자원 ↑).
  - GPU 사용 시 `WHISPER_DEVICE=cuda`, 정밀도 `float16` 등으로 변경. CPU-only 기본값은 `int8`로 메모리 사용을 줄임.
- Ollama:
  - entrypoint에서 모델을 선 pull 하므로 첫 실행 시 다운로드 시간이 필요; 프록시 환경에서는 사전 다운로드 혹은 수동 pull 필요.  
  - 추가 모델을 쓰려면 `entrypoint` 또는 수동 `ollama pull <model>` 추가.
- Qdrant:
  - 컬렉션 생성 시 임베딩 차원/거리(metric) 값을 `nomic-embed-text` 메타데이터와 일치시켜야 검색 품질 유지.  
  - 디스크 사용량은 `qdrant_data/`에서 직접 확인 가능.
- n8n:
  - `.env`의 `N8N_BASIC_AUTH_*`로 접근 보호.  
  - 워크플로우는 컨테이너 내부 `~/.n8n`에 저장되며 `n8n_data/`로 영속화됨.

## 인터페이스 예시
- STT:  
  ```bash
  curl -X POST "http://localhost:8000/stt" \
       -F "audio=@sample.wav" \
       -F "lang=ko"
  ```
- Ollama Chat:  
  ```bash
  curl -X POST http://localhost:11434/api/chat \
       -d '{ "model": "llama3", "messages": [{"role":"user","content":"회의 요약해줘"}] }'
  ```
- Ollama Embedding:  
  ```bash
  curl -X POST http://localhost:11434/api/embeddings \
       -d '{ "model": "nomic-embed-text", "input": "회의 내용 텍스트" }'
  ```
- Qdrant 검색:  
  ```bash
  curl -X POST http://localhost:6333/collections/notes/points/search \
       -H "Content-Type: application/json" \
       -d '{ "vector": [/* 임베딩 */], "limit": 3, "with_payload": true }'
  ```

## 운영 지표·트러블슈팅 메모
- 로깅: STT 서비스는 처리 시작/완료 로그와 예외 스택을 출력하므로 `docker-compose logs -f stt`로 지연/실패 상황 확인. n8n/ollama/qdrant도 동일하게 `logs -f <service>`로 모니터링.
- 전사 지연/정확도 문제:
  - CPU 환경에서 길거나 잡음 많은 음성은 느릴 수 있음 → 더 작은 모델 사용 또는 GPU 전환.  
  - 업로드 파일이 비어 있으면 400 반환(입력 유효성 확인).
- Ollama 초기화 지연: 첫 실행 시 모델 다운로드가 완료될 때까지 API 응답이 늦어질 수 있음. 필요 시 entrypoint 대기 시간을 늘리거나 사전 pull.
- Qdrant 스키마 불일치: 임베딩 차원/metric이 맞지 않으면 검색 점수가 왜곡됨 → 컬렉션 생성 시 모델 메타데이터 확인 후 일치시키기. 색인 재생성 시 `qdrant_data/` 백업 권장.
- 저장소/모델 크기: `ollama_data/`와 `qdrant_data/`가 디스크를 주로 사용하므로 로컬 용량 모니터링. Whisper 모델 크기와 Ollama 모델 크기(GB 단위)를 감안해 여유 공간 확보.
- 헬스체크: `curl http://localhost:8000/docs`, `http://localhost:11434`, `http://localhost:6333/metrics`, `http://localhost:5678` 등으로 서비스 기동 상태를 빠르게 검증.

## 이력서에 적기 좋은 한줄 포인트
- 로컬 전사(Whisper)→로컬 요약/임베딩(Ollama)→벡터 검색(Qdrant)→n8n 오케스트레이션으로 완전 오프라인 RAG 노트 테이커 구축.
- 컨테이너 간 데이터 흐름/스키마 정의, 모델 프리로드, 영속 볼륨 설계로 반복 실행 시 cold start와 데이터 손실 최소화.
- 워크플로우 예제(`workflows/*.md`)로 STT·요약·검색 시나리오를 재사용 가능하게 문서화.
