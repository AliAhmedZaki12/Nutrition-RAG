import os
import time
import numpy as np
import pandas as pd
from voyageai import Client
from dotenv import load_dotenv

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
MODEL = os.getenv("VOYAGE_MODEL", "voyage-3")

client = Client(api_key=VOYAGE_API_KEY)


def embed_chunks() -> np.ndarray:
  
    CSV_PATH = "backend/data/meta/chunks_meta.csv"
    EMB_PATH = "backend/data/embeddings/embeddings.npy"

   
    df = pd.read_csv(CSV_PATH)

    texts = df["sentence_chunk"].dropna().tolist()

    if not texts:
        raise ValueError(" No texts found for embedding")

    embeddings = []
    BATCH_SIZE = 32

    print(f"🔢 Embedding {len(texts)} chunks...")

    # 🔁 batching + retry
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        for attempt in range(3):
            try:
                r = client.embed(texts=batch, model=MODEL)
                embeddings.extend(r.embeddings)
                break
            except Exception as e:
                print(f" Retry {attempt + 1}: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(" Embedding failed after retries")

        time.sleep(0.5)

    
    arr = np.array(embeddings, dtype=np.float32)

    
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)

   
    np.save(EMB_PATH, arr)

    print(f" Embeddings saved → {EMB_PATH} | shape {arr.shape}")

    return arr
