# 📄 Multi-Modal Document Intelligence (RAG-Based QA System)

## 🚀 Project Overview

Modern real-world documents such as financial reports, policy papers (e.g., IMF Article IV reports), and enterprise contracts contain **text, tables, charts, figures, scanned images, and footnotes**. Traditional text-only QA systems fail to capture insights from such heterogeneous data.

This project implements a **Multi-Modal Retrieval-Augmented Generation (RAG) system** that can accurately answer user questions by jointly reasoning over **text, tables, and images (via OCR)** with proper **source citations**.

---

## 🎯 Key Objectives

* Build a **multi-modal document ingestion pipeline** (text, tables, images, OCR)
* Design a **smart chunking strategy** for complex document layouts
* Create a **unified embedding space** for multi-modal data
* Implement a **vector-based retrieval system**
* Develop a **QA chatbot** that produces **context-grounded, citation-backed answers**

---

## 🧠 System Architecture (High-Level)

```
PDF / Document
   ↓
Multi-Modal Ingestion
(text + tables + images + OCR)
   ↓
Smart Chunking
(semantic + layout-aware)
   ↓
Embeddings (Text & Vision)
   ↓
Vector Database (ChromaDB)
   ↓
Retriever + Re-ranking
   ↓
LLM (Groq / LLaMA / Mixtral)
   ↓
Answer + Page-Level Citations
   ↓
Streamlit / FastAPI UI
```

---

## 🧩 Core Components

### 1️⃣ Document Ingestion

* **Text Extraction:** PyMuPDF (fitz)
* **Table Parsing:** Row/column-aware structured extraction
* **Image Handling:** Image extraction + preprocessing with Pillow
* **OCR:** Converts scanned pages and figures into searchable text
* **Metadata:** Page number, section, modality preserved

### 2️⃣ Smart Chunking Strategy

* Semantic chunking (meaning-based)
* Layout-aware segmentation (page, section, table boundaries)
* Multi-modal alignment (text + table + OCR text)

Each chunk contains:

* Content
* Modality type (text / table / image)
* Source metadata (page, section)

### 3️⃣ Embedding Strategy

* **Text & Tables:** Sentence-Transformers / HuggingFace embeddings
* **Images:** OCR text + image metadata embeddings
* **Unified Vector Space:** Enables cross-modal retrieval

### 4️⃣ Vector Store & Retrieval

* **Vector DB:** ChromaDB
* **Retrieval:** Top-k semantic similarity search
* **Advanced (Bonus):**

  * Cross-modal reranking
  * Hybrid retrieval (RRF)

### 5️⃣ RAG-Based Answer Generation

* **Framework:** LangChain
* **LLMs:** Groq-hosted LLaMA / Mixtral
* **Output:**

  * Faithful answers grounded in retrieved context
  * Page or section-level citations

### 6️⃣ QA Interface

* **UI:** Streamlit / FastAPI
* Upload documents
* Ask natural language questions
* View answers with cited sources

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM Orchestration:** LangChain
* **LLMs:** Groq (LLaMA / Mixtral)
* **Embeddings:** Sentence-Transformers, HuggingFace
* **Vector Database:** ChromaDB
* **PDF Processing:** PyMuPDF
* **Image Processing:** Pillow
* **OCR:** Integrated OCR pipeline
* **Config Management:** python-dotenv

---

## 📦 Installation

```bash
# Clone repository
git clone <repo-url>
cd multi-modal-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Or (if using FastAPI):

```bash
uvicorn main:app --reload
```

---

## 📊 Evaluation & Benchmarking

* Benchmark queries across:

  * Text-based questions
  * Table-driven numerical queries
  * Image / OCR-based questions
* Metrics tracked:

  * Retrieval relevance
  * Answer faithfulness
  * Latency

---

## ✨ Bonus Features (Excellence Track)

* Cross-modal reranking using vision-text embeddings
* Hybrid retrieval (RRF)
* Summarization / briefing generation
* Retrieval & latency evaluation dashboard

---

## 📁 Deliverables

* ✅ Modular, well-documented codebase
* ✅ Interactive QA demo
* ✅ Technical report (architecture & findings)
* ✅ Video demonstration (3–5 mins)

---



---

## 👤 Author

**Hardik Somani**
Data Science & GenAI Engineer
📍 Rajasthan, India

---

## 📜 License

This project is for educational and demonstration purposes.
