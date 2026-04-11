from ingestion.ingest_pdf import ingest_pdf
from embedding.embed_chunks import embed_chunks
from vectorstore.pinecone_client import upsert_embeddings

PDF_URL = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"


def run_pipeline() -> None:
    print("\n🚀 STARTING RAG PIPELINE\n")

    ingest_pdf(
        pdf_path="data/raw/nutrition.pdf",
        download_url=PDF_URL,
        chunk_size=8,
    )

    embed_chunks()
    upsert_embeddings()

    print("\n✅ PIPELINE COMPLETE\n")


if __name__ == "__main__":
    run_pipeline()
