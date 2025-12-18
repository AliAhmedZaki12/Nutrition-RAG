import os
import requests
import fitz  # PyMuPDF
import pandas as pd
from tqdm.auto import tqdm

from utils import (
    text_formatter,
    split_sentences_spacy,
    create_sentence_chunks,
    filter_chunks
)


# 1. Download PDF if missing
def download_pdf(url: str, pdf_path: str):
    """
    Downloads a PDF from a URL if it is not already present.
    The style and messaging match the notebook's approach.
    """
    if not os.path.exists(pdf_path):
        print("File doesn't exist, downloading...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded PDF → {pdf_path}")
        else:
            raise RuntimeError(f"Failed to download file. Status code: {response.status_code}")
    else:
        print(f"File '{pdf_path}' already exists.")


# 2. Read PDF → page dictionary list
def open_and_read_pdf(pdf_path: str):
    """
    Reads a PDF file page-by-page with PyMuPDF.
    Extracts text, cleans it, and computes basic statistics.
    This mirrors the notebook's logic closely.
    
    Returns:
        list[dict] → each dict contains page_number, text, and stats
    """
    doc = fitz.open(pdf_path)
    pages_and_texts = []

    for page_number, page in tqdm(enumerate(doc)):
        # Extract raw text
        text = page.get_text()

        # Notebook-style cleaning
        text = text_formatter(text)

        page_dict = {
            "page_number": page_number,  
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4,  # approx: 1 token ~ 4 chars
            "text": text
        }
        pages_and_texts.append(page_dict)

    return pages_and_texts



# 3. Apply spaCy sentence splitting
def add_sentences_to_pages(pages_and_texts):
    """
    Adds 'sentences' and 'sentence_count_spacy' to each page.
    """
    for item in tqdm(pages_and_texts):
        sentences = split_sentences_spacy(item["text"])
        item["sentences"] = sentences
        item["page_sentence_count_spacy"] = len(sentences)
    return pages_and_texts



# 4. Convert sentence groups → chunks
def build_chunks_from_pages(pages_and_texts, sentence_chunk_size=10):
    """
    For each page, chunks its sentences into groups of N (default 10).
    Each chunk receives metadata and RAG-ready stats.
    """
    pages_and_chunks = []

    for item in tqdm(pages_and_texts):
        page_number = item["page_number"]
        sentences = item["sentences"]

        chunks = create_sentence_chunks(
            sentences=sentences,
            page_number=page_number,
            chunk_size=sentence_chunk_size
        )

        pages_and_chunks.extend(chunks)

    return pages_and_chunks


# 5. Entry point: Ingest entire PDF
def ingest_pdf(
    pdf_path: str,
    download_url: str = None,
    chunk_size: int = 10,
    min_token_length: int = 30,
    save_parquet: str = "chunks.parquet"
):
    """
    Full notebook-style ingestion pipeline:
    - Downloads PDF (if URL given)
    - Reads PDF
    - Splits pages into sentences
    - Splits sentences into chunks (size=chunk_size)
    - Filters tiny chunks (<min_token_length)
    - Saves final chunks to parquet for embedding
    
    Returns list of dicts (ready for embedding)
    """

    # Step 1 — download PDF
    if download_url:
        download_pdf(download_url, pdf_path)

    # Step 2 — read the text & compute stats
    print("\n Reading PDF...")
    pages = open_and_read_pdf(pdf_path)

    # Step 3 — sentence splitting
    print("\n Splitting text into sentences...")
    pages = add_sentences_to_pages(pages)

    # Step 4 — build chunks
    print("\n Building sentence chunks...")
    pages_and_chunks = build_chunks_from_pages(
        pages_and_texts=pages,
        sentence_chunk_size=chunk_size
    )

    # Step 5 — filter small chunks
    print("\n Filtering tiny chunks...")
    filtered_chunks = filter_chunks(pages_and_chunks, min_token_length=min_token_length)

    # Step 6 — save to parquet
    print(f"\n Saving chunks → {save_parquet}")
    df = pd.DataFrame(filtered_chunks)
    df.to_parquet(save_parquet, index=False)

    print(f" Done {len(filtered_chunks)} usable chunks created.")

    return filtered_chunks



# For standalone usage
if __name__ == "__main__":
    # Example PDF (you can replace with your own)
    pdf_path = "human-nutrition-text.pdf"
    url = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"

    ingest_pdf(
        pdf_path=pdf_path,
        download_url=url,
        chunk_size=10,
        min_token_length=30,
        save_parquet="chunks.parquet"
    )
