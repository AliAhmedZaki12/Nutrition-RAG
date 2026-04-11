def hybrid_search(
    dense_results: list[dict],
    sparse_results: list[dict],
    alpha: float = 0.7,
    top_k: int = 5,
) -> list[dict]:
    """
    Weighted fusion of dense (semantic) and sparse (BM25) results.
    alpha=0.7 → 70 % dense, 30 % sparse (tunable via env or caller).
    """
    combined: dict = {}

    for r in dense_results:
        key = (r["text"], r.get("page"))
        combined[key] = combined.get(key, 0.0) + alpha * r["score"]

    for r in sparse_results:
        key = (r["text"], r.get("page"))
        combined[key] = combined.get(key, 0.0) + (1 - alpha) * r["score"]

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [{"text": k[0], "page": k[1], "score": v} for k, v in ranked[:top_k]]
