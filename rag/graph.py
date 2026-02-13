"""LangGraph-based Agentic RAG pipeline.

Implements:
- Query decomposition: complex queries are split into sub-queries
- Conditional branching: route to retrieval or direct answer
- Iterative re-retrieval: if context is insufficient, reformulate and retry
- Reflection loop: self-evaluate answer quality, retry if needed
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from rag.config import LLM_MODEL, OLLAMA_BASE_URL, TOP_K_RERANK
from rag.retriever import RetrievedDoc, hybrid_search

logger = logging.getLogger("rag.graph")

MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------
class RAGState(BaseModel):
    """State that flows through the LangGraph pipeline."""

    original_query: str = ""
    sub_queries: list[str] = Field(default_factory=list)
    current_query: str = ""
    retrieved_docs: list[dict] = Field(default_factory=list)
    context: str = ""
    answer: str = ""
    is_relevant: bool = False
    is_faithful: bool = False
    retry_count: int = 0
    route: str = ""  # "retrieve" or "direct"


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------
def _get_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Node: Query Analysis & Decomposition
# ---------------------------------------------------------------------------
def analyze_query(state: RAGState) -> dict:
    """Analyze the query: decide routing and decompose if complex."""
    llm = _get_llm()

    # Routing decision
    route_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a query router. Given a user question, determine if it "
                "requires retrieving information from a knowledge base or can be "
                "answered directly.\n"
                "Respond with ONLY one word: 'retrieve' or 'direct'.\n"
                "- 'retrieve': questions about specific meetings, notes, documents, "
                "or factual information that would be stored.\n"
                "- 'direct': simple greetings, general knowledge, or meta-questions.",
            ),
            ("human", "{query}"),
        ]
    )
    route_chain = route_prompt | llm | StrOutputParser()
    route_result = route_chain.invoke({"query": state.original_query}).strip().lower()
    route = "retrieve" if "retrieve" in route_result else "direct"

    if route == "direct":
        return {"route": "direct", "current_query": state.original_query}

    # Query decomposition for complex queries
    decompose_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a query decomposition assistant. Given a complex question, "
                "break it down into 1-3 simpler sub-questions that together cover the "
                "original question.\n"
                "If the question is already simple, return it as-is.\n"
                "Return each sub-question on a new line. No numbering or bullets.\n"
                "Respond in the same language as the input question.",
            ),
            ("human", "{query}"),
        ]
    )
    decompose_chain = decompose_prompt | llm | StrOutputParser()
    raw_subs = decompose_chain.invoke({"query": state.original_query})
    sub_queries = [q.strip() for q in raw_subs.strip().split("\n") if q.strip()]

    if not sub_queries:
        sub_queries = [state.original_query]

    logger.info("Route=%s, sub_queries=%s", route, sub_queries)
    return {
        "route": route,
        "sub_queries": sub_queries,
        "current_query": sub_queries[0],
    }


# ---------------------------------------------------------------------------
# Node: Retrieve
# ---------------------------------------------------------------------------
def retrieve(state: RAGState) -> dict:
    """Retrieve documents using hybrid search for all sub-queries."""
    all_docs: dict[str, RetrievedDoc] = {}

    queries = state.sub_queries if state.sub_queries else [state.current_query]
    for query in queries:
        results = hybrid_search(query, top_k=TOP_K_RERANK)
        for doc in results:
            if doc.id not in all_docs or doc.score > all_docs[doc.id].score:
                all_docs[doc.id] = doc

    # Sort by score and deduplicate
    sorted_docs = sorted(all_docs.values(), key=lambda d: d.score, reverse=True)
    doc_dicts = [
        {"id": d.id, "text": d.text, "score": d.score, "metadata": d.metadata}
        for d in sorted_docs[:TOP_K_RERANK]
    ]

    # Build context string
    context_parts = []
    for i, doc in enumerate(doc_dicts, 1):
        context_parts.append(f"[Document {i}]\n{doc['text']}")
    context = "\n\n".join(context_parts)

    logger.info("Retrieved %d unique docs from %d sub-queries", len(doc_dicts), len(queries))
    return {"retrieved_docs": doc_dicts, "context": context}


# ---------------------------------------------------------------------------
# Node: Generate Answer
# ---------------------------------------------------------------------------
def generate(state: RAGState) -> dict:
    """Generate an answer using retrieved context."""
    llm = _get_llm(temperature=0.1)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers questions based on the "
                "provided context. Use ONLY the information from the context below. "
                "If the context doesn't contain enough information to answer, say so. "
                "Answer in the same language as the question.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{query}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke(
        {"context": state.context, "query": state.original_query}
    )
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Node: Direct Answer (no retrieval)
# ---------------------------------------------------------------------------
def direct_answer(state: RAGState) -> dict:
    """Answer directly without retrieval."""
    llm = _get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the user's question concisely. "
                "Answer in the same language as the question.",
            ),
            ("human", "{query}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"query": state.original_query})
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Node: Reflection / Self-evaluation
# ---------------------------------------------------------------------------
def reflect(state: RAGState) -> dict:
    """Evaluate the quality of the generated answer.

    Checks:
    1. Relevance: Does the answer address the original question?
    2. Faithfulness: Is the answer grounded in the retrieved context?
    """
    llm = _get_llm()

    eval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an answer quality evaluator. Given a question, context, "
                "and an answer, evaluate:\n"
                "1. RELEVANT: Does the answer address the question? (yes/no)\n"
                "2. FAITHFUL: Is the answer grounded in the context? (yes/no)\n\n"
                "Respond with exactly two lines:\n"
                "RELEVANT: yes/no\n"
                "FAITHFUL: yes/no",
            ),
            (
                "human",
                "Question: {query}\n\nContext: {context}\n\nAnswer: {answer}",
            ),
        ]
    )
    chain = eval_prompt | llm | StrOutputParser()
    result = chain.invoke(
        {
            "query": state.original_query,
            "context": state.context,
            "answer": state.answer,
        }
    )

    is_relevant = "yes" in result.lower().split("relevant:")[-1].split("\n")[0]
    is_faithful = "yes" in result.lower().split("faithful:")[-1].split("\n")[0]

    logger.info(
        "Reflection: relevant=%s, faithful=%s, retry=%d/%d",
        is_relevant,
        is_faithful,
        state.retry_count,
        MAX_RETRIES,
    )
    return {
        "is_relevant": is_relevant,
        "is_faithful": is_faithful,
        "retry_count": state.retry_count + 1,
    }


# ---------------------------------------------------------------------------
# Node: Reformulate Query (for re-retrieval)
# ---------------------------------------------------------------------------
def reformulate(state: RAGState) -> dict:
    """Reformulate the query when reflection indicates poor quality."""
    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "The previous retrieval did not produce a satisfactory answer. "
                "Reformulate the following question to improve search results. "
                "Use different keywords or rephrase the question. "
                "Return only the reformulated question. "
                "Respond in the same language as the input.",
            ),
            ("human", "Original: {original}\nPrevious attempt: {current}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    new_query = chain.invoke(
        {"original": state.original_query, "current": state.current_query}
    )
    logger.info("Reformulated query: %s", new_query.strip())
    return {
        "current_query": new_query.strip(),
        "sub_queries": [new_query.strip()],
    }


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------
def route_after_analysis(state: RAGState) -> Literal["retrieve", "direct_answer"]:
    return "direct_answer" if state.route == "direct" else "retrieve"


def route_after_reflection(
    state: RAGState,
) -> Literal["end", "reformulate"]:
    if state.is_relevant and state.is_faithful:
        return "end"
    if state.retry_count >= MAX_RETRIES:
        logger.warning("Max retries reached, returning best answer")
        return "end"
    return "reformulate"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_rag_graph() -> StateGraph:
    """Build the Agentic RAG graph with LangGraph.

    Flow:
    analyze_query → [route]
      ├── direct_answer → END
      └── retrieve → generate → reflect → [quality check]
                                             ├── good → END
                                             └── bad → reformulate → retrieve (loop)
    """
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("direct_answer", direct_answer)
    graph.add_node("reflect", reflect)
    graph.add_node("reformulate", reformulate)

    # Set entry point
    graph.set_entry_point("analyze_query")

    # Conditional routing after analysis
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "retrieve": "retrieve",
            "direct_answer": "direct_answer",
        },
    )

    # Linear edges
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "reflect")
    graph.add_edge("direct_answer", END)

    # Conditional after reflection
    graph.add_conditional_edges(
        "reflect",
        route_after_reflection,
        {
            "end": END,
            "reformulate": "reformulate",
        },
    )

    # Reformulate loops back to retrieve
    graph.add_edge("reformulate", "retrieve")

    return graph


# Compiled graph (singleton)
rag_graph = build_rag_graph().compile()


def run_rag(query: str) -> dict:
    """Execute the full Agentic RAG pipeline.

    Returns dict with: answer, retrieved_docs, metadata.
    """
    initial_state = RAGState(original_query=query, current_query=query)
    final_state = rag_graph.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "retrieved_docs": final_state.get("retrieved_docs", []),
        "route": final_state.get("route", ""),
        "retry_count": final_state.get("retry_count", 0),
        "is_relevant": final_state.get("is_relevant", None),
        "is_faithful": final_state.get("is_faithful", None),
    }
