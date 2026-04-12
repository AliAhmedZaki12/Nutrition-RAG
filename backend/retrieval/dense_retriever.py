import os

NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")


def dense_search(index, query_embedding: list, top_k: int = 5) -> list[dict]:
    res = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
    )

    results = []

    for m in res.matches:
        results.append(
            {
                "id": m.metadata.get("id"),
                "text": m.metadata.get("text", ""),
                "page": m.metadata.get("page", "-"),
                "score": m.score,
            }
        )

    return results
