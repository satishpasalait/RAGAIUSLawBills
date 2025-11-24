# RAG over US Law Bills (Python + FastAPI)

This project is a **Retrieval-Augmented Generation (RAG) REST API** built in Python.
It allows you to ask natural-language questions over real US legislative bills using:

- Local embeddings (Sentence Transformers)
- A vector database (ChromaDB)
- A REST API (FastAPI)
- An LLM for answering (OpenAI – optional / swappable)

---

## Features

- Ingests a public dataset of US Congressional bills  
- Chunks + embeds the text into a vector database  
- Performs semantic search over the data  
- Uses retrieved context to generate grounded AI answers  
- Exposes everything via a REST API endpoint  
- Uses **local embeddings** (no cost / no rate limits)

---

## Project Structure

RAGAIUSLawBills/
│
├── ingest.py # Ingests dataset and builds vector index
├── app.py # FastAPI RAG service
├── bill_sum_data.csv # Dataset file
├── chroma_store/ # ChromaDB vector database folder
├── .env # API keys (optional if using OpenAI)
└── README.md


---

## Prerequisites

- Python **3.10+**
- pip
- Virtual environment recommended

### Create Virtual Environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate

Mac / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
