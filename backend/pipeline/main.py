from backend.ingestion.ingest_pdf import ingest_pdf
from backend.embedding.embed_chunks import embed_chunks
from backend.vectorstore.pinecone_client import upsert_embeddings
import os

PDF_URL  = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"
RAW_PATH = "backend/data/raw/nutrition.pdf"
CSV_PATH = "backend/data/meta/chunks_meta.csv"
EMB_PATH = "backend/data/embeddings/embeddings.npy"


def run_pipeline(force: bool = False) -> None:
    print("\n🚀 STARTING RAG PIPELINE\n")

    # ─────────────────────────────
    # 1. Ingestion
    # ─────────────────────────────
    if force or not os.path.exists(CSV_PATH):
        print("📄 Running ingestion...")
        ingest_pdf(pdf_path=RAW_PATH, download_url=PDF_URL, chunk_size=8)
    else:
        print("⏩ Skipping ingestion (already exists)")

    # ─────────────────────────────
    # 2. Embedding
    # ─────────────────────────────
    if force or not os.path.exists(EMB_PATH):
        print("🔢 Running embedding...")
        embed_chunks()
    else:
        print("⏩ Skipping embedding (already exists)")

    # ─────────────────────────────
    # 3. Pinecone Upload
    # ✅ FIX — تمرير force للـ upsert حتى لا يُعاد الرفع في كل مرة
    # ─────────────────────────────
    print("📤 Checking Pinecone...")
    upsert_embeddings(force=force)   # ✅ كان: upsert_embeddings() بدون force
    print("\n✅ PIPELINE COMPLETE\n")


if __name__ == "__main__":
    run_pipeline(force=False)
