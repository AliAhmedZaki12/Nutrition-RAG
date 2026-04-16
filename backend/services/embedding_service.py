import os
import time
import numpy as np
from functools import lru_cache
from voyageai import Client
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────
# ✅ FIX — Client كان يُنشأ على module level
#
# المشكلة القديمة:
#   VOYAGE_API_KEY = os.getenv(...)
#   client = Client(api_key=VOYAGE_API_KEY)   ← module level
#   → لو VOYAGE_API_KEY = None وقت الـ import → crash فوري
#
# الحل:
#   lazy initialization — الـ client يتعمل أول مرة محتاجينه بس
# ─────────────────────────────────────────────────────

VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3")
_client: Client | None = None


def _get_client() -> Client:
    """Lazy init — يتنفذ بعد تحميل الـ .env بشكل مضمون."""
    global _client
    if _client is None:
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("❌ VOYAGE_API_KEY is not set in environment")
        _client = Client(api_key=api_key)
    return _client


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    client     = _get_client()
    embeddings: list = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        for attempt in range(3):
            try:
                r = client.embed(texts=batch, model=VOYAGE_MODEL)
                embeddings.extend(r.embeddings)
                break
            except Exception as e:
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError("❌ Embedding failed after retries")

    return np.array(embeddings, dtype=np.float32)


@lru_cache(maxsize=256)
def cached_embed(query: str) -> list[float]:
    """Cache repeated queries to avoid redundant API calls."""
    return embed_texts([query])[0].tolist()
