"""
LangGraph 기반 Agentic RAG 파이프라인.

노드 구성:
  analyse_query → route
    ├─ (direct) → generate_direct → END
    └─ (retrieve) → retrieve → generate → reflect
                                             ├─ (accept) → END
                                             └─ (retry, max 2) → rewrite_query → retrieve → ...
"""

import logging
import os
from typing import Literal

import requests
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from rag.retriever import SearchResult, hybrid_search

logger = logging.getLogger("rag.graph")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict, total=False):
    query: str
    sub_queries: list[str]
    route: str  # "retrieve" | "direct"
    contexts: list[dict]
    answer: str
    reflection: str
    is_acceptable: bool
    retry_count: int
    rewritten_query: str


# ---------------------------------------------------------------------------
# LLM 호출 헬퍼
# ---------------------------------------------------------------------------

def _llm_chat(system: str, user: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ---------------------------------------------------------------------------
# 노드: 쿼리 분석 (라우팅 + 분해)
# ---------------------------------------------------------------------------

def analyse_query(state: RAGState) -> RAGState:
    query = state["query"]
    logger.info("Analysing query: %s", query[:100])

    system = (
        "You are a query analyzer. Given the user question, decide:\n"
        "1. route: 'retrieve' if the question needs information from stored documents, 'direct' if it can be answered from general knowledge.\n"
        "2. sub_queries: If the question is complex, break it into 1-3 simpler sub-queries. If simple, return only the original.\n"
        "Respond EXACTLY in this format (no extra text):\n"
        "ROUTE: retrieve\n"
        "QUERIES:\n"
        "- sub query 1\n"
        "- sub query 2"
    )
    result = _llm_chat(system, query)

    route = "retrieve"
    if "ROUTE: direct" in result:
        route = "direct"

    sub_queries = []
    in_queries = False
    for line in result.splitlines():
        line = line.strip()
        if line.startswith("QUERIES:"):
            in_queries = True
            continue
        if in_queries and line.startswith("- "):
            sub_queries.append(line[2:].strip())

    if not sub_queries:
        sub_queries = [query]

    logger.info("Route=%s sub_queries=%s", route, sub_queries)
    return {**state, "route": route, "sub_queries": sub_queries, "retry_count": 0}


# ---------------------------------------------------------------------------
# 라우팅 함수
# ---------------------------------------------------------------------------

def route_decision(state: RAGState) -> Literal["retrieve", "generate_direct"]:
    return "retrieve" if state.get("route") == "retrieve" else "generate_direct"


# ---------------------------------------------------------------------------
# 노드: 직접 답변 (검색 불필요)
# ---------------------------------------------------------------------------

def generate_direct(state: RAGState) -> RAGState:
    query = state["query"]
    answer = _llm_chat(
        "You are a helpful assistant. Answer the question directly.",
        query,
    )
    return {**state, "answer": answer, "contexts": []}


# ---------------------------------------------------------------------------
# 노드: 검색
# ---------------------------------------------------------------------------

def retrieve(state: RAGState) -> RAGState:
    sub_queries = state.get("sub_queries", [state["query"]])
    if state.get("rewritten_query"):
        sub_queries = [state["rewritten_query"]]

    all_contexts: list[dict] = []
    seen_ids: set[str] = set()
    for sq in sub_queries:
        results: list[SearchResult] = hybrid_search(sq, top_k=3)
        for r in results:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                all_contexts.append({"id": r.id, "text": r.text, "score": r.score})

    logger.info("Retrieved %d unique contexts", len(all_contexts))
    return {**state, "contexts": all_contexts}


# ---------------------------------------------------------------------------
# 노드: 생성
# ---------------------------------------------------------------------------

def generate(state: RAGState) -> RAGState:
    query = state["query"]
    contexts = state.get("contexts", [])
    context_str = "\n\n---\n\n".join(c["text"] for c in contexts)

    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "If the context doesn't contain enough information, say so. Cite relevant parts."
    )
    user_msg = f"Context:\n{context_str}\n\nQuestion: {query}"
    answer = _llm_chat(system, user_msg)
    return {**state, "answer": answer}


# ---------------------------------------------------------------------------
# 노드: 반성 (관련성·충실성 자체 평가)
# ---------------------------------------------------------------------------

def reflect(state: RAGState) -> RAGState:
    query = state["query"]
    answer = state.get("answer", "")
    contexts = state.get("contexts", [])
    context_str = "\n".join(c["text"] for c in contexts[:3])

    system = (
        "You are a strict evaluator. Check:\n"
        "1. Is the answer relevant to the question?\n"
        "2. Is the answer faithful to the provided context (no hallucination)?\n"
        "Respond EXACTLY:\n"
        "RELEVANT: yes/no\n"
        "FAITHFUL: yes/no\n"
        "REASON: one line explanation"
    )
    user_msg = f"Question: {query}\nAnswer: {answer}\nContext: {context_str}"
    result = _llm_chat(system, user_msg)

    is_relevant = "RELEVANT: yes" in result
    is_faithful = "FAITHFUL: yes" in result
    is_acceptable = is_relevant and is_faithful

    logger.info("Reflection: relevant=%s faithful=%s acceptable=%s", is_relevant, is_faithful, is_acceptable)
    return {
        **state,
        "reflection": result,
        "is_acceptable": is_acceptable,
    }


def reflect_decision(state: RAGState) -> Literal["end", "rewrite_query"]:
    if state.get("is_acceptable", False):
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        logger.info("Max retries reached, accepting current answer")
        return "end"
    return "rewrite_query"


# ---------------------------------------------------------------------------
# 노드: 쿼리 재구성
# ---------------------------------------------------------------------------

def rewrite_query(state: RAGState) -> RAGState:
    query = state["query"]
    reflection = state.get("reflection", "")
    retry_count = state.get("retry_count", 0) + 1

    rewritten = _llm_chat(
        "Rewrite the following question to get better search results. "
        "Consider the evaluation feedback. Return only the rewritten query, nothing else.",
        f"Original: {query}\nFeedback: {reflection}",
    )
    logger.info("Rewritten query (retry %d): %s", retry_count, rewritten[:100])
    return {**state, "rewritten_query": rewritten.strip(), "retry_count": retry_count}


# ---------------------------------------------------------------------------
# 그래프 구성
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("analyse_query", analyse_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("generate_direct", generate_direct)
    graph.add_node("reflect", reflect)
    graph.add_node("rewrite_query", rewrite_query)

    graph.set_entry_point("analyse_query")
    graph.add_conditional_edges("analyse_query", route_decision, {
        "retrieve": "retrieve",
        "generate_direct": "generate_direct",
    })
    graph.add_edge("generate_direct", END)
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "reflect")
    graph.add_conditional_edges("reflect", reflect_decision, {
        "end": END,
        "rewrite_query": "rewrite_query",
    })
    graph.add_edge("rewrite_query", "retrieve")

    return graph


rag_graph = build_graph().compile()
