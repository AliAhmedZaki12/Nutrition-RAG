"""
RAG Retrieval Pipeline

Steps:
1. Embed query using Voyage AI
2. Retrieve similar chunks from Pinecone
3. Build RAG prompt
4. Send prompt to LLM
"""

import os
import numpy as np
from dotenv import load_dotenv
from voyageai import Client
from pinecone import Pinecone

# Local helpers
from utils import prompt_formatter
from llm_openrouter import generate_answer   # <-- IMPORTANT

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL = os.getenv("VOYAGE_MODEL", "voyage-3")

if not PINECONE_API_KEY or not VOYAGE_API_KEY:
    raise ValueError("Missing Pinecone/Voyage API keys in .env")

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
voyage = Client(api_key=VOYAGE_API_KEY)


# ---------------------------------------------------------
# 1. Embed query using Voyage AI
# ---------------------------------------------------------

def embed_query(query: str):
    response = voyage.embed(texts=[query], model=VOYAGE_MODEL)
    embedding = np.array(response.embeddings[0], dtype=np.float32)
    return embedding.tolist()


# ---------------------------------------------------------
# 2. Retrieve top-k from Pinecone
# ---------------------------------------------------------

def retrieve(query: str, top_k: int = 5):
    print(f"\n🔍 Query: {query}")

    q_emb = embed_query(query)

    results = index.query(
        vector=q_emb,
        top_k=top_k,
        include_metadata=True
    )

    contexts = []
    for match in results.matches:
        meta = match.metadata
        contexts.append({
            "text": meta.get("sentence_chunk", ""),
            "page": meta.get("page_number", "unknown"),
            "score": match.score
        })

    return contexts


# ---------------------------------------------------------
# 3. Build the RAG prompt
# ---------------------------------------------------------

def build_rag_prompt(query: str, top_k: int = 5):
    contexts = retrieve(query, top_k=top_k)
    prompt = prompt_formatter(query, contexts)
    return prompt, contexts


# ---------------------------------------------------------
# 4. Run full RAG pipeline (Retrieve → Prompt → LLM Answer)
# ---------------------------------------------------------

def rag_answer(query: str, top_k: int = 5):
    prompt, contexts = build_rag_prompt(query, top_k)

    print("\n===== CONTEXTS =====")
    for c in contexts:
        print(f"[Page {c['page']}] (Score={c['score']:.4f})")
        print(c["text"], "\n")

    print("\n===== FINAL PROMPT =====")
    print(prompt)

    print("\n===== LLM ANSWER =====")
    answer = generate_answer(prompt)
    print(answer)

    return answer


# ---------------------------------------------------------
# Manual test
# ---------------------------------------------------------

if __name__ == "__main__":
    rag_answer("macronutrients functions", top_k=4)
