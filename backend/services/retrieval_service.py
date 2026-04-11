from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor   # ✅ FIX #4

from backend.utils.prompt_formatter import prompt_formatter
from backend.retrieval.dense_retriever import dense_search
from backend.retrieval.sparse_retriever import SparseRetriever
from backend.retrieval.hybrid import hybrid_search
from backend.llm.llm_openrouter import generate_answer
from backend.services.embedding_service import cached_embed

sparse: SparseRetriever | None = None
dense_index = None


# ─────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────

def init_retrievers(index, documents: list[dict]) -> None:
    global sparse, dense_index
    sparse = SparseRetriever(documents)
    dense_index = index
    print("✅ Retrievers initialized")


# ─────────────────────────────────────────────────────
# ✅ FIX #5 — Context Compression
# ─────────────────────────────────────────────────────

_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "of", "and", "or", "for", "it", "this", "that", "with", "be", "by",
}


def _query_tokens(query: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\b\w+\b", query)} - _STOP_WORDS


def compress_context(query: str, chunks: list[dict], max_sentences: int = 3) -> list[dict]:
    """
    Keep only the most query-relevant sentences from each chunk.
    Reduces prompt length → lower cost, better focus, fewer hallucinations.
    """
    q_tokens = _query_tokens(query)
    compressed = []
    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])
        scored = sorted(
            sentences,
            key=lambda s: len({w.lower() for w in re.findall(r"\b\w+\b", s)} & q_tokens),
            reverse=True,
        )
        compressed.append({**chunk, "text": " ".join(scored[:max_sentences])})
    return compressed


# ─────────────────────────────────────────────────────
# ✅ FIX #6 — Smart Chunk Selection (MMR-lite)
# ─────────────────────────────────────────────────────

def _trigrams(text: str) -> frozenset:
    words = text.lower().split()
    return frozenset(tuple(words[i : i + 3]) for i in range(len(words) - 2))


def smart_chunk_selection(chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    MMR-lite: select high-score chunks while skipping near-duplicates.
    If > 60 % of a chunk's trigrams already appear in selected chunks → drop it.
    """
    selected: list[dict] = []
    seen: frozenset = frozenset()

    for chunk in sorted(chunks, key=lambda x: x["score"], reverse=True):
        tg = _trigrams(chunk["text"])
        if selected and tg:
            overlap = len(tg & seen) / len(tg)
            if overlap > 0.60:
                continue
        selected.append(chunk)
        seen = seen | tg
        if len(selected) >= top_k:
            break

    return selected


# ─────────────────────────────────────────────────────
# Main service
# ─────────────────────────────────────────────────────

def rag_answer_hybrid_service(query: str, top_k: int = 5) -> tuple[str, list[dict]]:
    if sparse is None or dense_index is None:
        raise RuntimeError("Retrievers not initialized. Run the pipeline first.")

    q_emb = cached_embed(query)

    # ✅ FIX #4 — parallel dense + sparse search (~40 % lower latency)
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future  = executor.submit(dense_search, dense_index, q_emb, top_k)
        sparse_future = executor.submit(sparse.search, query, top_k)
        dense_results  = dense_future.result()
        sparse_results = sparse_future.result()

    # Fuse results (wider pool before filtering)
    hybrid = hybrid_search(dense_results, sparse_results, alpha=0.7, top_k=top_k * 2)

    # ✅ FIX #6 — remove near-duplicate chunks
    smart = smart_chunk_selection(hybrid, top_k=top_k)

    # ✅ FIX #5 — compress each chunk to most relevant sentences
    compressed = compress_context(query, smart, max_sentences=3)

    prompt = prompt_formatter(query, compressed)
    answer = generate_answer(prompt)
    return answer, compressed
