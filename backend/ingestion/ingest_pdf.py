import os
import requests
import fitz  # PyMuPDF
import pandas as pd
from backend.ingestion.utils import (
    text_formatter,
    split_sentences_spacy,
    create_sentence_chunks,
    filter_chunks,
)

# ─────────────────────────────────────────────────────
# ✅ FIX #2 — توحيد المسارات
# القديم: "data/parquet/" و "data/meta/"  ← خاطئ
# الجديد: "backend/data/parquet/" و "backend/data/meta/"  ← صح
# ─────────────────────────────────────────────────────

PARQUET_DIR = "backend/data/parquet"
META_DIR    = "backend/data/meta"


def download_pdf(url: str, pdf_path: str) -> None:
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    if not os.path.exists(pdf_path):
        print("⬇️  Downloading PDF...")
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            with open(pdf_path, "wb") as f:
                f.write(r.content)
            print("✅ PDF downloaded")
        else:
            raise RuntimeError(f"Download failed: {r.status_code}")


def load_chunks(parquet_path: str) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    return df.to_dict(orient="records")


def ingest_pdf(
    pdf_path: str,
    download_url: str = None,
    chunk_size: int = 8,
) -> list[dict]:

    if download_url and not os.path.exists(pdf_path):
        download_pdf(download_url, pdf_path)

    print("📄 Reading PDF...")
    doc    = fitz.open(pdf_path)
    pages  = []

    for i, page in enumerate(doc):
        text      = text_formatter(page.get_text())
        sentences = split_sentences_spacy(text)
        chunks    = create_sentence_chunks(sentences, i, chunk_size)
        pages.extend(chunks)

    chunks = filter_chunks(pages)
    df     = pd.DataFrame(chunks)

    # ✅ FIX — حفظ في المسارات الصح
    os.makedirs(PARQUET_DIR, exist_ok=True)
    os.makedirs(META_DIR,    exist_ok=True)

    df.to_parquet(f"{PARQUET_DIR}/chunks.parquet", index=False)
    df.to_csv(f"{META_DIR}/chunks_meta.csv",       index=False)

    print(f"✅ Ingestion complete: {len(chunks)} chunks")
    return chunks
