"""watsonx.ai 공통 클라이언트 — 생성(LLM) + 임베딩.

RAG 파이프라인 전체가 Gemini 대신 IBM watsonx.ai를 사용하도록 통일한다.
  - 생성: meta-llama/llama-3-3-70b-instruct (chat)
  - 임베딩: intfloat/multilingual-e5-large (1024차원, 한국어 multilingual)
"""

import logging
import os

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings, ModelInference

logger = logging.getLogger("rag.watsonx")

WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY", "")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "")

GEN_MODEL = os.getenv("WATSONX_GEN_MODEL", "meta-llama/llama-3-3-70b-instruct")
EMBED_MODEL = os.getenv("WATSONX_EMBED_MODEL", "intfloat/multilingual-e5-large")

# 임베딩 차원 (Qdrant 컬렉션 크기와 반드시 일치해야 함)
EMBED_DIM = int(os.getenv("WATSONX_EMBED_DIM", "1024"))

_credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_APIKEY)

_gen_model = ModelInference(
    model_id=GEN_MODEL,
    credentials=_credentials,
    project_id=WATSONX_PROJECT_ID,
    params={
        "max_new_tokens": 1024,
        "temperature": 0,
        "decoding_method": "greedy",
    },
)

_embed_model = Embeddings(
    model_id=EMBED_MODEL,
    credentials=_credentials,
    project_id=WATSONX_PROJECT_ID,
)


def chat(system: str, user: str) -> str:
    """system/user 메시지로 watsonx chat 호출 후 텍스트 반환."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    resp = _gen_model.chat(messages=messages)
    return resp["choices"][0]["message"]["content"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 리스트의 임베딩 벡터 리스트 반환."""
    return _embed_model.embed_documents(texts=texts)


def embed_query(text: str) -> list[float]:
    """단일 텍스트의 임베딩 벡터 반환."""
    return _embed_model.embed_documents(texts=[text])[0]
