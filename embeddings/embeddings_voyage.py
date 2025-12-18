import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm
from voyageai import Client
import time


load_dotenv()


VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3")

if VOYAGE_API_KEY is None:
    raise ValueError("Missing VOYAGE_API_KEY in .env file")

client = Client(api_key=VOYAGE_API_KEY)



# 2. Helper: batch embed

def embed_texts(text_list, model, batch_size=32):
    """
    Embeds texts using Voyage AI with built-in throttling
    for free-tier rate limits (3 RPM, 10K TPM).

    Adds a 20-second sleep between each batch.
    """

    all_embeddings = []

    total = len(text_list)

    for i in tqdm(range(0, total, batch_size), desc="Embedding chunks"):
        batch = text_list[i : i + batch_size]

        try:
            response = client.embed(
                texts=batch,
                model=model
            )
        except Exception as e:
            print(f"\n Voyage API error: {e}")
            print(" Waiting 30 seconds before retrying...")
            time.sleep(30)
            # Retry once
            response = client.embed(
                texts=batch,
                model=model
            )

        all_embeddings.extend(response.embeddings)

        print("⏳ Sleeping 20 seconds to respect Voyage free-tier limits...")
        time.sleep(20)

    return all_embeddings


# 3. Main embedding workflow

def embed_chunks(
    parquet_path="chunks.parquet",
    embeddings_out="embeddings.npy",
    meta_out="chunks_meta.csv"
):
    """
    Loads chunks from parquet, embeds them with Voyage AI, and saves:
    - embeddings.npy : matrix of embeddings
    - chunks_meta.csv : text metadata (no vectors)
    """

    print("📦 Loading chunks from:", parquet_path)
    df = pd.read_parquet(parquet_path)

    # The column from ingest_pdf.py is "sentence_chunk"
    if "sentence_chunk" not in df.columns:
        raise ValueError("Expected column 'sentence_chunk' not found in chunks parquet file.")

    texts = df["sentence_chunk"].tolist()

    print(f" Embedding {len(texts)} chunks with Voyage AI ({VOYAGE_MODEL})")
    embeddings = embed_texts(texts, model=VOYAGE_MODEL, batch_size=32)

    # Convert to float32 matrix
    embeddings = np.array(embeddings).astype(np.float32)

    print(f" Saving embeddings → {embeddings_out}")
    np.save(embeddings_out, embeddings)

    # Save metadata WITHOUT embeddings
    print(f" Saving metadata → {meta_out}")
    df.to_csv(meta_out, index=False)

    print(" Embedding complete!")
    print(f"Total embeddings shape: {embeddings.shape}")

    return embeddings


# If running standalone

if __name__ == "__main__":
    embed_chunks(
        parquet_path="chunks.parquet",
        embeddings_out="embeddings.npy",
        meta_out="chunks_meta.csv"
    )
