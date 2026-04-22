# 🥗 NutriAI — Production-Grade Hybrid RAG Nutrition Assistant

**Live Demo:**
[https://nutri-ai--aliahmedzaki788.replit.app](https://nutri-ai--aliahmedzaki788.replit.app)

---

## 🚀 Overview

**NutriAI** is a production-grade AI system that generates **personalised 4-week meal plans** using a **Hybrid Retrieval-Augmented Generation (RAG)** pipeline.

Unlike traditional LLM chatbots, NutriAI:

* Retrieves information from a **verified academic nutrition textbook**
* Applies **hybrid search (semantic + keyword)**
* Generates responses **grounded in real evidence**

> **Core Principle:** Retrieval First. Generation Second.

---

## 🎯 Problem Statement

LLM-only systems fail in health-related domains due to:

* ❌ Hallucinated medical advice
* ❌ Inconsistent answers
* ❌ No source of truth

### ✅ Solution

NutriAI enforces:

* Deterministic retrieval
* Verified knowledge source
* Controlled generation

---

## 🧠 System Architecture

```
Frontend (SPA)
      │
      ▼
FastAPI (Single Deployment)
      │
      ▼
Hybrid RAG Pipeline
      │
      ▼
LLM (OpenRouter + Fallback Chain)
```

### Key Design Decision

> ✅ **Single Service Deployment (Frontend + Backend)**

* No CORS issues
* No multi-service complexity
* Faster iteration & debugging

---

## 🔄 RAG Pipeline

```
User Query + Profile
        │
        ▼
Embedding (Voyage AI)
        │
        ▼
Parallel Retrieval
 ├── Dense → Pinecone
 └── Sparse → BM25
        │
        ▼
Hybrid Fusion (RRF)
        │
        ▼
MMR Deduplication
        │
        ▼
Context Compression
        │
        ▼
Prompt Construction
        │
        ▼
LLM Generation
        │
        ▼
Final Answer
```

---

## ⚙️ Key Features

### 🔍 Hybrid Retrieval

* Dense + Sparse fusion
* Reciprocal Rank Fusion (RRF)
* Adaptive retrieval (dynamic `top_k`)

### ⚡ Performance Optimization

* LRU embedding cache
* Parallel retrieval (ThreadPool)
* ~60% token reduction via compression

### 🛡️ Reliability

* Multi-model fallback chain
* Graceful degradation
* Strict input validation (Pydantic)

### 🧍 Personalization

* User profile injection:

  * Age, weight, conditions
  * Goals & preferences
  * Daily meal context

---

## 🌐 Frontend–Backend Integration

```javascript
var API_BASE = window.location.origin;
```

**Why this matters:**

* Eliminates hardcoded URLs
* Prevents CORS issues
* Works across all environments (local / Replit / production)

---

## 📊 Performance

| Metric             | Value     |
| ------------------ | --------- |
| End-to-end latency | ~1.2s     |
| Retrieval time     | 150–300ms |
| Embedding (cached) | 30–50ms   |
| Token reduction    | ~60%      |
| Pinecone vectors   | 1906      |
| Uptime             | ~99.9%    |

---

## 🧠 Engineering Challenges & Solutions

### 🔌 API Design

**Problem:** Mismatch between frontend (POST) and backend (GET)
**Solution:** Unified API with Pydantic validation
**Impact:** Eliminated 405 / 422 errors

---

### ⚙️ Startup Failures

**Problem:** System returned 503 despite running
**Cause:** Silent import failure
**Solution:** Explicit readiness state
**Impact:** Predictable system behavior

---

### 🌐 CORS & Deployment Issues

**Problem:** Cross-origin blocking + broken URLs
**Solution:** Single-origin architecture + dynamic base URL
**Impact:** Zero CORS complexity

---

### 🧠 Weak Retrieval Quality

**Problem:** Low similarity → poor answers
**Solution:** Adaptive retrieval expansion
**Impact:** Higher recall + better answers

---

### 🧾 Redundant Context

**Problem:** Duplicate chunks
**Solution:** MMR-lite deduplication
**Impact:** More diverse context

---

### 🧍 Missing Personalization

**Problem:** Profile ignored
**Solution:** Injected into prompt pipeline
**Impact:** Personalised responses

---

### 📦 Pinecone SDK Changes

**Problem:** Breaking API changes
**Solution:** Version-safe access layer
**Impact:** Stable integration

---

### 🎨 UI Issues

**Problems:**

* Markdown not rendered
* Meal cards clipped

**Solutions:**

* Markdown → HTML parser
* CSS fixes

**Impact:** Clean UI/UX

---

### 🍽️ Meal Plan Repetition

**Problem:** Identical weekly plans
**Solution:**

* Thematic diversity (4 cuisines)
* 100+ unique meals

**Impact:** Realistic meal planning

---

## 🧱 Tech Stack

| Layer         | Technology       |
| ------------- | ---------------- |
| Backend       | FastAPI          |
| Frontend      | Vanilla JS (SPA) |
| Embeddings    | Voyage AI        |
| Vector DB     | Pinecone         |
| Sparse Search | BM25             |
| LLM           | OpenRouter       |
| Data          | Pandas / NumPy   |
| Deployment    | Replit           |

---

## 📁 Project Structure

```
NutriAI/
│
├── Frontend/
│   └── index.html
│
└── backend/
    ├── main.py
    ├── routes.py
    ├── services/
    ├── retrieval/
    ├── pipeline/
    ├── llm/
    ├── utils/
    └── vectorstore/
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
pip install -r backend/requirements.txt
```

---

### 2. Set environment variables

```
PINECONE_API_KEY=
VOYAGE_API_KEY=
OPENROUTER_API_KEY=
```

---

### 3. Run the app

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 5000
```

---

### 4. Verify

```
GET /status → ready
GET / → UI loads
```

---

## 🔌 API Reference

### POST `/query`

```json
{
  "q": "What foods help hypertension?",
  "top_k": 5,
  "profile": {...}
}
```

### Response

```json
{
  "answer": "...",
  "context": [...],
  "chunks_used": 5
}
```

---

## 📚 Data Pipeline

* Source: Human Nutrition (OER Hawaii)
* ~500 pages
* 3000+ chunks
* 1024-d embeddings
* Indexed in Pinecone

---

## 🧠 Design Principles

* **Deterministic > Probabilistic**
* **Simple systems scale better**
* **Fail gracefully**
* **Minimize hallucination**

---

## 🚧 Future Work

* Streaming responses (SSE/WebSockets)
* Query classification (skip RAG when unnecessary)
* RAG evaluation (RAGAS)
* User memory layer
* Multi-source knowledge base

---

## 👨‍💻 Author

**Eng. Ali Zaki**
AI Engineer — RAG Systems & Applied LLMs
