def hybrid_search(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    

    scores = {}

    # Dense
    for rank, r in enumerate(dense_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + 1 / (k + rank)

    # Sparse
    for rank, r in enumerate(sparse_results):
        rid = r["id"]
        scores[rid] = scores.get(rid, 0) + 1 / (k + rank)

    # Merge metadata
    merged = {}

    for r in dense_results + sparse_results:
        rid = r["id"]
        if rid not in merged:
            merged[rid] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {**merged[rid], "score": score}
        for rid, score in ranked[:top_k]
    ]
