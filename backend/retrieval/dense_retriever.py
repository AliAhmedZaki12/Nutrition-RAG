import os

NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")


def dense_search(index, query_embedding: list, top_k: int = 5) -> list[dict]:
    """
    Query Pinecone with the correct namespace.
    ✅ FIX #1: namespace was missing → all queries returned 0 results.
    """
    res = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,        # ← critical fix
    )
    results = []
    for m in res.matches:
        results.append(
            {
                "text": m.metadata.get("sentence_chunk", ""),
                "page": m.metadata.get("page_number", "-"),
                "score": m.score,
            }
        )
    return results
