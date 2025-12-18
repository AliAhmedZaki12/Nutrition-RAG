# 🥗 Nutrition-RAG: PDF-Based Intelligent Nutrition Assistant

<img width="2000" height="600" alt="image" src="https://github.com/user-attachments/assets/acd0df9c-23fd-466a-ae7e-60c17ca79d2c" />

An end-to-end **Retrieval-Augmented Generation (RAG)** project that allows you to chat with a nutrition textbook PDF using modern GenAI tools. This project is beginner-friendly and walks you through building a RAG system **from scratch**, exactly as it was developed step by step.

---

##  Project Overview

This project enables users to:
- Upload and index a nutrition PDF
- Retrieve relevant text chunks using semantic search
- Generate accurate answers grounded in the document
- Interact through a clean **Streamlit GUI**

The system combines **Voyage AI embeddings**, **Pinecone vector database**, and **Meta LLaMA 3.1 (8B)** via **OpenRouter**.

---

##  Tech Stack

###  Embedding Model
- **Voyage AI**
  - Model: `voyage-3`
  - Used to convert text chunks and user queries into dense vectors

###  Vector Database
- **Pinecone**
  - Stores and retrieves embeddings using similarity search
  - Cloud-hosted, scalable, and fast

###  Large Language Model (LLM)
- **meta-llama/llama-3.1-8b-instruct**
  - Accessed via **OpenRouter**
  - Generates answers strictly based on retrieved context

###  Backend
- Python
- NumPy
- Requests
- dotenv

###  Frontend
- **Streamlit** (interactive web-based UI)

---

##  Project Structure

```text
nutrition-rag-project/
│
├── ingestion.py           # PDF loading, chunking, embedding, Pinecone upsert
├── retrieval.py           # Query embedding + Pinecone similarity search
├── llm_openrouter.py      # OpenRouter LLM call logic
├── utils.py               # Prompt formatting helpers
├── chat.py                # Streamlit GUI
├── requirements.txt
├── .env                   # API keys (NOT committed)
└── README.md
```
##  Step 1: Create a Pinecone Index

<p align="left">
  <a href="https://www.pinecone.io" target="_blank">
    <img src="https://img.shields.io/badge/Pinecone-Vector%20Database-0A0A0A?style=for-the-badge&logo=pinecone&logoColor=white"/>
  </a>
</p>

1. Create a **new index**
   - **Index name**: `nutrition-rag-project`
   - **Dimensions**: `1024` (required for `voyage-3`)
   - **Metric**: `cosine`
2. Once created, copy your **Pinecone API Key**
3. Keep the index name exactly the same — it will be used in the code


---

##  Step 2: Set Up OpenRouter (LLM Gateway)

<p align="left">
  <a href="https://openrouter.ai" target="_blank">
    <img src="https://img.shields.io/badge/OpenRouter-LLM%20Gateway-1F2937?style=for-the-badge&logo=openai&logoColor=white"/>
  </a>
</p>

1. Create an OpenRouter account  
2. Add a **billing method** (required even for free models — no charge)  
3. Generate an **API Key**
4. LLM used in this project:
   - `meta-llama/llama-3.1-8b-instruct`

> OpenRouter automatically selects an available provider for the model.


---

##  Step 3: Set Up Voyage AI (Embeddings)

<p align="left">
  <a href="https://www.voyageai.com" target="_blank">
    <img src="https://img.shields.io/badge/Voyage%20AI-Embeddings-2563EB?style=for-the-badge&logo=vectorworks&logoColor=white"/>
  </a>
</p>

1. Create a Voyage AI account  
2. Generate a **Voyage API Key**
3. Embedding model used:
   - `voyage-3`
   - Output dimension: `1024`


---


##  Step 4: Environment Variables

This repository includes an environment template file:

```text
.env.example   # API keys template (NO real keys)
````

### What you must do:

1. Copy `.env.example`
2. Rename it to `.env`
3. Replace the placeholder values with your real API keys

### Example `.env` file:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=nutrition-rag-project

VOYAGE_API_KEY=your_voyage_api_key
VOYAGE_MODEL=voyage-3

OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
```
---

## Step 5: Install Dependencies

Install all required Python libraries using:

```bash
pip install -r requirements.txt
```

Make sure you are using **Python 3.10+**.

---

## Step 6: Ingest the PDF (One-Time Setup)

This step prepares your knowledge base. It will:

* Download the nutrition PDF (if not present)
* Clean and normalize text
* Split text into sentence-based chunks
* Generate embeddings using **Voyage AI**
* Upload vectors + metadata to **Pinecone**

Run **once only**:

```bash
python ingestion.py
```

 After this step, your Pinecone index is fully ready.

---

##  Step 7: Test Retrieval + RAG Pipeline

(Optional but highly recommended)

This step verifies that everything works end-to-end:

* Query embedding
* Pinecone similarity search
* Prompt construction
* LLM response generation

Run:

```bash
python retrieval.py
```

If you see retrieved chunks and a grounded answer, your RAG pipeline is correct.

---

##  Step 8: Run the Streamlit Chat Application


```bash
streamlit run chat.py
```

---

##  Example Question

```text
What are the functions of macronutrients?
```



## Tech Stack Summary

*  **LLM:** meta-llama/llama-3.1-8b-instruct
*  **Embeddings:** voyage-3
*  **Vector Database:** Pinecone
*  **LLM Gateway:** OpenRouter
*  **Frontend:** Streamlit
*  **Language:** Python

