# app/retrieval.py

import os
import sys
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from voyageai import Client
import pinecone

# ---------------------------------------------------------
# Add project root to sys.path (necessary for Streamlit/Cloud)
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
from utils import prompt_formatter
from llm.llm_openrouter import generate_llm_answer

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3")

if not PINECONE_API_KEY or not VOYAGE_API_KEY:
    raise ValueError("Missing Pinecone or Voyage API keys in environment variables.")

# ---------------------------------------------------------
# Initialize clients
# ---------------------------------------------------------
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX_NAME)

voyage = Client(api_key=VOYAGE_API_KEY)

# ---------------------------------------------------------
# Embed query
# ---------------------------------------------------------
def embed_query(query: str):
    """Embed a query string using Voyage AI"""
    response = voyage.embed(texts=[query], model=VOYAGE_MODEL)
    embedding = np.array(response.embeddings[0], dtype=np.float32)
    return embedding.tolist()

# ---------------------------------------------------------
# Retrieve top-k relevant chunks from Pinecone
# ---------------------------------------------------------
def retrieve(query: str, top_k: int = 5):
    """Retrieve top-k similar chunks for a query"""
    q_emb = embed_query(query)
    results = index.query(vector=q_emb, top_k=top_k, include_metadata=True)

    contexts = [
        {
            "text": match.metadata.get("sentence_chunk", ""),
            "page": match.metadata.get("page_number", "unknown"),
            "score": match.score
        }
        for match in results.matches
    ]
    return contexts

# ---------------------------------------------------------
# Build RAG prompt
# ---------------------------------------------------------
def build_rag_prompt(query: str, top_k: int = 5):
    """Build prompt for LLM using retrieved chunks"""
    contexts = retrieve(query, top_k=top_k)
    prompt = prompt_formatter(query, contexts)
    return prompt, contexts

# ---------------------------------------------------------
# Full RAG answer
# ---------------------------------------------------------
def rag_answer(query: str, top_k: int = 5, max_tokens: int = 512, temperature: float = 0.2):
    """Retrieve → Build prompt → LLM answer"""
    prompt, contexts = build_rag_prompt(query, top_k)
    answer = generate_llm_answer(prompt, max_tokens=max_tokens, temperature=temperature)
    return answer

# ---------------------------------------------------------
# Manual test
# ---------------------------------------------------------
if __name__ == "__main__":
    test_query = "Functions of macronutrients"
    print("Query:", test_query)
    answer = rag_answer(test_query, top_k=4)
    print("Answer:", answer)

