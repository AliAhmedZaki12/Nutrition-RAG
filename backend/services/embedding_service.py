import os
import time
import numpy as np
from functools import lru_cache
from voyageai import Client
from dotenv import load_dotenv

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3")

client = Client(api_key=VOYAGE_API_KEY)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    embeddings: list = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

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
