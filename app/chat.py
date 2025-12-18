# chat.py
import sys
from pathlib import Path

# -------------------------------------------------
# Ensure project root is in PYTHONPATH (Streamlit Cloud safe)
# -------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# -------------------------------------------------
# Project imports
# -------------------------------------------------
from vectorstore.retrieval import build_rag_prompt
from llm.llm_openrouter import generate_llm_answer

import streamlit as st

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Nutrition-RAG | Intelligent PDF Assistant",
    page_icon="🥗",
    layout="wide"
)

# -------------------------------------------------
# Custom CSS (clean & professional)
# -------------------------------------------------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #2C7BE5;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #6c757d;
            margin-bottom: 1.5rem;
        }
        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #1f2937;
            margin-top: 1.2rem;
        }
        .answer-box {
            background-color: #f8f9fa;
            border-left: 5px solid #2C7BE5;
            padding: 1rem;
            border-radius: 0.5rem;
            font-size: 1.05rem;
            color: #1f2937;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("⚙️ Configuration")

top_k = st.sidebar.slider(
    "🔍 Retrieved Chunks (Top-K)",
    min_value=1,
    max_value=10,
    value=4,
)

temperature = st.sidebar.slider(
    "🎛️ Model Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05,
)

max_tokens = st.sidebar.slider(
    "✍️ Max Answer Tokens",
    min_value=128,
    max_value=2048,
    value=512,
    step=64,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **System Stack**
    - 🧠 **LLM:** meta-llama/llama-3.1-8b-instruct  
    - 📐 **Embeddings:** Voyage-3  
    - 🗂️ **Vector DB:** Pinecone  
    - 🌐 **Interface:** Streamlit
    """
)

# -------------------------------------------------
# Main UI
# -------------------------------------------------
st.markdown(
    '<div class="main-title">🥗 Nutrition-RAG Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Ask grounded questions from your nutrition PDF using Retrieval-Augmented Generation.'
    '</div>',
    unsafe_allow_html=True,
)

# -------------------------------------------------
# User input
# -------------------------------------------------
user_query = st.text_input(
    "🔎 Enter your question",
    placeholder="e.g. What are the functions of macronutrients?",
)

ask_clicked = st.button("🚀 Ask Question")

# -------------------------------------------------
# RAG Pipeline
# -------------------------------------------------
if ask_clicked:
    if not user_query.strip():
        st.warning("Please enter a valid question.")
    else:
        # -----------------------------
        # Retrieval
        # -----------------------------
        with st.spinner("Retrieving relevant document chunks..."):
            prompt, context_chunks = build_rag_prompt(
                user_query,
                top_k=top_k,
            )

        st.markdown(
            '<div class="section-title">📚 Retrieved Context</div>',
            unsafe_allow_html=True,
        )

        for i, c in enumerate(context_chunks, start=1):
            with st.expander(
                f"Chunk {i} | Page {c.get('page', 'N/A')} | "
                f"Score: {c.get('score', 0.0):.4f}"
            ):
                st.write(c.get("text", ""))

        # -----------------------------
        # LLM Answer
        # -----------------------------
        st.markdown(
            '<div class="section-title">🤖 Model Answer</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Generating answer using the LLM..."):
            answer = generate_llm_answer(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True,
        )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption(
    "Nutrition-RAG | Built with Streamlit, Voyage AI, Pinecone, and OpenRouter"
)
