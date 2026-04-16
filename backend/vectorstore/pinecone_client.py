import os
import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────
# ✅ FIX A — pc = Pinecone() كان على module level
#   المشكلة: لو PINECONE_API_KEY = None وقت الـ import → crash
#   الحل: lazy init داخل دالة _get_pc()
#
# ✅ FIX B — i["name"] فاشل في Pinecone SDK v3+
#   المشكلة: pc.list_indexes() بترجع objects مش dicts
#   → i["name"] بيطلع TypeError: 'IndexModel' not subscriptable
#   الحل: استخدام i.name (attribute access)
#
# ✅ FIX C — upsert_embeddings() كانت تعيد رفع كل البيانات في كل مرة
#   المشكلة: حتى لو البيانات موجودة في Pinecone → وقت وأموال ضايعة
#   الحل: إضافة فحص total_vector_count قبل الرفع
# ─────────────────────────────────────────────────────

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "nutrition-rag-project")
NAMESPACE  = os.getenv("PINECONE_NAMESPACE",  "default")
CLOUD      = os.getenv("PINECONE_CLOUD",       "aws")
REGION     = os.getenv("PINECONE_REGION",      "eu-west-1")

EMB_PATH = "backend/data/embeddings/embeddings.npy"
CSV_PATH = "backend/data/meta/chunks_meta.csv"

_pc: Pinecone | None = None


def _get_pc() -> Pinecone:
    """✅ FIX A — Lazy init بعد تحميل الـ .env."""
    global _pc
    if _pc is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("❌ PINECONE_API_KEY is not set in environment")
        _pc = Pinecone(api_key=api_key)
    return _pc


def get_index():
    return _get_pc().Index(INDEX_NAME)


def upsert_embeddings(force: bool = False) -> None:
    pc         = _get_pc()
    embeddings = np.load(EMB_PATH)
    df         = pd.read_csv(CSV_PATH)

    if len(embeddings) != len(df):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings vs {len(df)} rows"
        )

    dim = embeddings.shape[1]

    # ✅ FIX B — .name بدل ["name"] لـ SDK v3+
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
        print(f"✅ Created index '{INDEX_NAME}' (dim={dim})")

    index = pc.Index(INDEX_NAME)

    # ✅ FIX C — تحقق من وجود بيانات قبل الرفع
    if not force:
        stats       = index.describe_index_stats()
        total_vecs  = stats.get("total_vector_count", 0)
        if total_vecs >= len(df):
            print(f"⏩ Skipping upsert — {total_vecs} vectors already in Pinecone")
            return

    BATCH_SIZE = 100
    print(f"📤 Uploading {len(df)} vectors to Pinecone...")

    for i in range(0, len(df), BATCH_SIZE):
        batch = []
        for j in range(min(BATCH_SIZE, len(df) - i)):
            idx = i + j
            row = df.iloc[idx]
            batch.append({
                "id":     str(row["id"]),
                "values": embeddings[idx].tolist(),
                "metadata": {
                    "id":   str(row["id"]),
                    "text": row["sentence_chunk"],
                    "page": int(row["page_number"]),
                },
            })
        index.upsert(vectors=batch, namespace=NAMESPACE)

    stats = index.describe_index_stats()
    print(f"✅ Done. Total vectors: {stats['total_vector_count']}")
