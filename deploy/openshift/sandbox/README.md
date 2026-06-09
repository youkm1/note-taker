# OpenShift Sandbox 배포 가이드

Red Hat Developer Sandbox(`youkm0806-dev`) 기준.  
Qdrant는 Qdrant Cloud, STT는 RTZR API를 사용해 Sandbox 7Gi 쿼터 안에 동작.

## 예상 메모리 사용량

| 서비스 | requests | limits |
|--------|----------|--------|
| STT    | 128Mi    | 512Mi  |
| RAG    | 1Gi      | 3Gi    |
| Eval   | 256Mi    | 1Gi    |
| **합계** | **~1.4Gi** | **~4.5Gi** |

Sandbox 7Gi 쿼터에 여유 있음.

## 사전 준비 — 가지고 있어야 할 값

| 값 | 출처 |
|----|------|
| `GEMINI_API_KEY` | Google AI Studio |
| `GEMINI_API_KEY_EVAL` | Google AI Studio (같은 키 재사용 가능) |
| `QDRANT_API_KEY` | Qdrant Cloud → API Keys |
| Qdrant Cloud URL | Qdrant Cloud → Cluster 상세 (`:6333` 포트 붙이기) |
| `RTZR_CLIENT_ID` / `RTZR_CLIENT_SECRET` | developers.rtzr.ai |
| `IBM_STT_APIKEY` / `IBM_STT_URL` | 벤치마크용, 없으면 빈 문자열 |

## 배포 절차

### 1. oc 로그인

OpenShift 웹 콘솔 우측 상단 `youkm0806 ▼` → **로그인 명령 복사** → 터미널에 붙여넣기

```bash
oc login --token=<token> --server=<server>
oc project youkm0806-dev
```

### 2. ConfigMap — Qdrant Cloud URL 수정

`01-configmap.yaml` 에서 `QDRANT_URL` 값을 실제 Qdrant Cloud URL로 교체:

```yaml
QDRANT_URL: "https://b64012b7-xxxx.eu-west-1-0.aws.cloud.qdrant.io:6333"
```

> ⚠️ `:6333` 포트 필수. 없으면 RAG가 연결 못 함.

### 3. Secret 생성

```bash
oc create secret generic notetaker-secrets \
  --from-literal=GEMINI_API_KEY='...' \
  --from-literal=GEMINI_API_KEY_EVAL='...' \
  --from-literal=QDRANT_API_KEY='...' \
  --from-literal=RTZR_CLIENT_ID='...' \
  --from-literal=RTZR_CLIENT_SECRET='...' \
  --from-literal=IBM_STT_APIKEY='' \
  --from-literal=IBM_STT_URL=''
```

### 4. 매니페스트 적용 (빌드 자동 시작)

```bash
oc apply -k deploy/openshift/sandbox
```

ConfigChange 트리거로 BuildConfig가 즉시 빌드를 시작.

### 5. 빌드 / 파드 상태 확인

```bash
# 빌드 로그 실시간 확인
oc logs -f bc/notetaker-rag
oc logs -f bc/notetaker-stt
oc logs -f bc/notetaker-eval

# 파드 상태 (빌드 완료 후 Running이 되어야 함)
oc get pods

# Route URL 확인
oc get routes
```

빌드는 서비스당 3~8분 소요 (첫 번째 빌드, 레이어 캐시 없음).

### 6. 동작 확인

```bash
# oc get routes 에서 나온 실제 URL로 교체
STT_URL=https://notetaker-stt-youkm0806-dev.apps.<cluster>.openshiftapps.com
RAG_URL=https://notetaker-rag-youkm0806-dev.apps.<cluster>.openshiftapps.com
EVAL_URL=https://notetaker-eval-youkm0806-dev.apps.<cluster>.openshiftapps.com

curl $STT_URL/health
curl $RAG_URL/health
curl $EVAL_URL/health

# OpenAPI spec (watsonx Orchestrate 임포트용)
curl $RAG_URL/openapi.json
```

### 재빌드 (코드 변경 후)

```bash
oc start-build notetaker-rag --follow
# 빌드 완료 → ImageStream 갱신 → Deployment 자동 롤링 업데이트
```

## 삭제

```bash
oc delete -k deploy/openshift/sandbox
oc delete secret notetaker-secrets
```
