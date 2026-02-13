"""RAGAS-based RAG quality evaluation pipeline.

Metrics:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevance: Does the answer address the question?
- Context Precision: Are the retrieved documents relevant to the question?
- Context Recall: Does the retrieved context cover the ground truth?

Usage:
  POST /evaluate       — evaluate a single QA pair
  POST /evaluate/batch — evaluate a dataset of QA pairs
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger("eval")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag:8001")


def _get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )


def _get_embeddings():
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def evaluate_single(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
) -> dict:
    """Evaluate a single QA pair using RAGAS metrics.

    Args:
        question: The user question.
        answer: The generated answer.
        contexts: Retrieved context documents.
        ground_truth: Optional reference answer for context_recall.

    Returns:
        Dictionary of metric scores.
    """
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    if ground_truth:
        data["ground_truth"] = [ground_truth]

    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy]
    if ground_truth:
        metrics.extend([context_precision, context_recall])

    llm = _get_llm()
    embeddings = _get_embeddings()

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    scores = {k: round(v, 4) for k, v in result.items() if isinstance(v, (int, float))}
    logger.info("Evaluation scores: %s", scores)
    return scores


def evaluate_batch(
    samples: list[dict],
) -> dict:
    """Evaluate a batch of QA pairs.

    Each sample dict should have:
      - question: str
      - answer: str (optional — will query RAG service if missing)
      - contexts: list[str] (optional — will query RAG service if missing)
      - ground_truth: str (optional)

    Returns aggregated metrics and per-sample scores.
    """
    questions = []
    answers = []
    all_contexts = []
    ground_truths = []
    has_ground_truth = False

    for sample in samples:
        question = sample["question"]

        # If answer or contexts missing, query the RAG service
        if "answer" not in sample or "contexts" not in sample:
            rag_result = _query_rag(question)
            answer = sample.get("answer", rag_result.get("answer", ""))
            contexts = sample.get(
                "contexts",
                [s.get("text", "") for s in rag_result.get("sources", [])],
            )
        else:
            answer = sample["answer"]
            contexts = sample["contexts"]

        questions.append(question)
        answers.append(answer)
        all_contexts.append(contexts)

        if "ground_truth" in sample:
            ground_truths.append(sample["ground_truth"])
            has_ground_truth = True
        else:
            ground_truths.append("")

    data = {
        "question": questions,
        "answer": answers,
        "contexts": all_contexts,
    }
    if has_ground_truth:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy]
    if has_ground_truth:
        metrics.extend([context_precision, context_recall])

    llm = _get_llm()
    embeddings = _get_embeddings()

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    scores = {k: round(v, 4) for k, v in result.items() if isinstance(v, (int, float))}
    logger.info("Batch evaluation (%d samples): %s", len(samples), scores)
    return {
        "aggregate": scores,
        "num_samples": len(samples),
    }


def _query_rag(question: str) -> dict:
    """Call the RAG service to get answer and sources."""
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{RAG_SERVICE_URL}/query",
            json={"question": question},
        )
        resp.raise_for_status()
    return resp.json()
