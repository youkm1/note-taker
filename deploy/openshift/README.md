# OpenShift Deployment Draft

This directory is a first OpenShift migration draft for the note-taker PoC.

The target interview story is:

```text
Docker Compose local PoC
  -> Red Hat OpenShift on IBM Cloud
  -> watsonx Orchestrate calls the exposed APIs as tools
```

## What Gets Deployed

- `stt`: STT API. Default provider stays configurable through `STT_PROVIDER`.
- `rag`: Agentic RAG API with LangGraph.
- `eval`: RAGAS evaluation API.
- `qdrant`: vector database with persistent storage.

`n8n` is intentionally not deployed here because the migration target is watsonx Orchestrate.

## Files

- `00-namespace.yaml`: namespace for the PoC.
- `01-configmap.yaml`: non-sensitive runtime config.
- `02-secret.template.yaml`: sensitive values template. Do not commit real values.
- `03-qdrant.yaml`: Qdrant PVC, Deployment, and Service.
- `04-stt.yaml`: STT Deployment and Service.
- `05-rag.yaml`: RAG Deployment and Service.
- `06-eval.yaml`: Eval Deployment and Service.
- `07-routes.yaml`: OpenShift Routes for public API access.
- `kustomization.yaml`: portable apply target after the secret is created.

## Before Applying

Build and push the images to IBM Cloud Container Registry or another registry that the OpenShift cluster can pull from.

Example image names:

```text
icr.io/<registry-namespace>/note-taker-stt:latest
icr.io/<registry-namespace>/note-taker-rag:latest
icr.io/<registry-namespace>/note-taker-eval:latest
```

Then update image names in:

- `04-stt.yaml`
- `05-rag.yaml`
- `06-eval.yaml`

## Create The Secret

Option A: edit and apply the template:

```bash
cp deploy/openshift/02-secret.template.yaml /tmp/notetaker-secret.yaml
oc apply -f /tmp/notetaker-secret.yaml
```

Option B: create it directly:

```bash
oc new-project note-taker
oc create secret generic notetaker-secrets \
  --from-literal=GEMINI_API_KEY='<rag-api-key>' \
  --from-literal=GEMINI_API_KEY_EVAL='<eval-api-key>'
```

Optional STT provider credentials can be added later:

```bash
oc patch secret notetaker-secrets \
  --type merge \
  -p '{"stringData":{"IBM_STT_APIKEY":"<apikey>","IBM_STT_URL":"<service-url>"}}'
```

## Deploy

```bash
oc apply -k deploy/openshift
```

Check status:

```bash
oc get pods
oc get svc
oc get routes
oc logs deploy/notetaker-rag
```

Health checks:

```bash
curl https://<stt-route-host>/health
curl https://<rag-route-host>/health
curl https://<eval-route-host>/health
```

## Why OpenShift For This Project

- `qdrant` needs persistent storage, so this is not a great fit for a purely stateless/serverless deployment.
- `stt`, `rag`, and `eval` are already containerized.
- Secrets, Services, PVCs, and Routes map cleanly to the current Docker Compose architecture.
- It matches IBM's Red Hat hybrid cloud story better than plain local Docker Compose.

