import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import pinecone  # ✅ Modern Pinecone import

load_dotenv()

# 1️⃣ Load configs
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "my-rag-index")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")  # Update based on your account

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY in .env")

# 2️⃣ Initialize Pinecone client
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)


# 3️⃣ Create or get index
def create_or_get_index(index_name: str, dimension: int):
    """
    Creates a Pinecone index if it doesn't exist.
    Otherwise returns the existing one.
    """
    existing_indexes = pinecone.list_indexes()

    if index_name not in existing_indexes:
        print(f"Creating Pinecone index '{index_name}' with dimension {dimension} ...")
        pinecone.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine"  # You can change to 'dotproduct' if needed
        )
        print("Index created!")
    else:
        print(f"Index '{index_name}' already exists.")

    # Return the live index (connect to it)
    return pinecone.Index(index_name)


# 4️⃣ Batch-upload embeddings
def upsert_embeddings(
    embeddings_file="embeddings.npy",
    metadata_file="chunks_meta.csv",
    batch_size=100
):
    """
    Loads `embeddings.npy` + `chunks_meta.csv` and inserts them into Pinecone in batches.
    """
    print("Loading embeddings & metadata...")
    embeddings = np.load(embeddings_file)  # shape: (N, dim)
    df = pd.read_csv(metadata_file)

    if len(df) != embeddings.shape[0]:
        raise ValueError("Mismatch: metadata rows and embedding rows are not equal!")

    dim = embeddings.shape[1]
    print(f"Embedding dimension detected: {dim}")

    # Create or load index
    index = create_or_get_index(PINECONE_INDEX_NAME, dim)

    print(f"Upserting {len(df)} vectors to Pinecone...")

    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i + batch_size]
        batch_vecs = embeddings[i:i + batch_size]

        to_upsert = []
        for j, row in batch_df.iterrows():
            vector_id = f"chunk-{i + j - i}"  # id = chunk-<row index>
            vector = batch_vecs[j - i].tolist()
            metadata = row.to_dict()
            to_upsert.append({
                "id": vector_id,
                "values": vector,
                "metadata": metadata
            })

        index.upsert(vectors=to_upsert)

    print("Upsert completed successfully")


# 5️⃣ Standalone execution
if __name__ == "__main__":
    upsert_embeddings(
        embeddings_file="embeddings.npy",
        metadata_file="chunks_meta.csv",
        batch_size=100
    )
