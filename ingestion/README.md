# PDF Ingestion & Text Utilities

This section explains **`ingest_pdf.py`** and **`utils.py`**, the two foundational files responsible for transforming raw PDF documents into clean, structured, and RAG-ready data.

These files operate **before embeddings, vector databases, or LLMs** and form the backbone of the entire pipeline.

---

## Overview

| File | Purpose |
|------|--------|
| **ingest_pdf.py** | Controls the document ingestion pipeline (PDF → chunks) |
| **utils.py** | Provides reusable text processing utilities (cleaning, splitting, chunking, prompting) |

**Key idea:**  
`ingest_pdf.py` orchestrates the workflow, while `utils.py` contains the reusable logic.

---

## ingest_pdf.py — PDF Ingestion Pipeline

### What this file does

`ingest_pdf.py` converts a raw PDF document into structured semantic chunks that are ready for embedding and retrieval.

Pipeline steps:
1. Download the PDF (if not available locally)
2. Read the PDF page-by-page
3. Clean extracted text
4. Split text into sentences
5. Group sentences into meaningful chunks
6. Remove tiny or noisy chunks
7. Save final chunks to disk (`parquet`)

---

## Pipeline Flow

```text
PDF
 ↓
Page-level text extraction
 ↓
Sentence splitting (spaCy)
 ↓
Sentence chunking
 ↓
Noise filtering
 ↓
chunks.parquet (RAG-ready)
```

---

### Key Functions

**`download_pdf()`**  
Downloads the PDF only if it doesn’t already exist, ensuring reproducibility.

**`open_and_read_pdf()`**  
Extracts text per page using PyMuPDF and computes page-level statistics (tokens, words, sentences).

**`add_sentences_to_pages()`**  
Applies linguistic sentence segmentation and adds sentence lists to each page.

**`build_chunks_from_pages()`**  
Groups sentences into paragraph-like chunks while preserving semantic meaning.

**`ingest_pdf()`**  
Orchestrates the full ingestion pipeline and saves the final output to `chunks.parquet`.

---

### Output

The output is a structured dataset where each row represents a meaningful text chunk with metadata such as:
- Page number  
- Clean chunk text  
- Token, word, and character statistics  

This file does **not** interact with embeddings, vector databases, or LLMs.

---

## utils.py — Text Processing & Prompt Utilities

### What this file does

`utils.py` contains all reusable helper functions related to text processing and prompt construction.  
It focuses on preparing clean, well-structured text and building grounded RAG prompts.

---

### Key Functions Explained

**`text_formatter()`**  
Cleans raw PDF text by removing line breaks and spacing artifacts.

**`split_sentences_spacy()`**  
Uses spaCy’s sentencizer for reliable sentence-level splitting.

**`create_sentence_chunks()`**  
Groups sentences into fixed-size semantic chunks and computes chunk-level statistics.

**`filter_chunks()`**  
Removes very small or irrelevant chunks such as headers and footers.

**`prompt_formatter()`**  
Builds a structured RAG prompt by injecting retrieved context and enforcing grounded, detailed answers without hallucination.

---

## Difference Between ingest_pdf.py and utils.py

| Aspect | ingest_pdf.py | utils.py |
|------|---------------|----------|
| Role | Pipeline controller | Utility provider |
| Responsibility | Orchestration | Text processing logic |
| Handles files | Yes | No |
| Reusable across projects | Limited | High |
| PDF awareness | Yes | No |
| RAG prompt logic | No | Yes |

---

## How They Work Together

```text
ingest_pdf.py
 ├─ calls text_formatter()
 ├─ calls split_sentences_spacy()
 ├─ calls create_sentence_chunks()
 └─ calls filter_chunks()

utils.py
 └─ provides all reusable text-processing and prompt-formatting logic
```
