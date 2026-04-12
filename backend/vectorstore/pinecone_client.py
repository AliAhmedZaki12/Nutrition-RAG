import os
import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "nutrition-rag-project")
NAMESPACE        = os.getenv("PINECONE_NAMESPACE",  "default")
CLOUD            = os.getenv("PINECONE_CLOUD",       "aws")
REGION           = os.getenv("PINECONE_REGION",      "eu-west-1")

pc = Pinecone(api_key=PINECONE_API_KEY)


def get_index():
    return pc.Index(INDEX_NAME)


def upsert_embeddings() -> None:
    embeddings = np.load("data/embeddings/embeddings.npy")
    df = pd.read_csv("data/meta/chunks_meta.csv")

    if len(embeddings) != len(df):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings vs {len(df)} rows"
        )

    dim = embeddings.shape[1]

    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        print(f"Created index '{INDEX_NAME}' (dim={dim})")

    index = pc.Index(INDEX_NAME)

    BATCH_SIZE = 100
    print("Uploading to Pinecone...")

    for i in range(0, len(df), BATCH_SIZE):
        batch = []

        for j in range(min(BATCH_SIZE, len(df) - i)):
            idx = i + j
            row = df.iloc[idx]

            batch.append(
                {
                    "id": str(row["id"]),
                    "values": embeddings[idx].tolist(),
                    "metadata": {
                        "id": str(row["id"]),
                        "text": row["sentence_chunk"],
                        "page": row["page_number"],
                    },
                }
            )

        index.upsert(vectors=batch, namespace=NAMESPACE)

    stats = index.describe_index_stats()
    print(f"Done. Total vectors: {stats}")
