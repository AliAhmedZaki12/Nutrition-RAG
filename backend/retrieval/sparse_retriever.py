import re
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """
    ✅ FIX #3: proper tokenization — lowercase + strip punctuation.
    Old split() was weak (kept punctuation, case-sensitive).
    This improves BM25 precision and recall significantly.
    """
    return re.findall(r"\b\w+\b", text.lower())


class SparseRetriever:
    def __init__(self, documents: list[dict]):
        self.texts = [doc["sentence_chunk"] for doc in documents]
        self.tokenized = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self.tokenized)
        self.docs = documents

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            doc = self.docs[idx]
            results.append(
                {
                    "text": doc["sentence_chunk"],
                    "page": doc["page_number"],
                    "score": float(score),
                }
            )
        return results
