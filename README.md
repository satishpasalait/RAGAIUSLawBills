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

``` bash
python -m venv .venv
.venv\Scripts\activate
```

Mac / Linux:

``` bash
python -m venv .venv
source .venv/bin/activate
```
### Install Dependencies

``` bash
pip install fastapi uvicorn[standard] openai chromadb python-dotenv pandas sentence-transformers
```

#### Create .env file if using OpenAI

```ini
OPENAI_API_KEY=your_openai_key_here
```

## Download Dataset
We use the BillSum public dataset of US Congressional bills:

```bash
curl https://raw.githubusercontent.com/Azure-Samples/Azure-OpenAI-Docs-Samples/main/Samples/Tutorials/Embeddings/data/bill_sum_data.csv --output bill_sum_data.csv
```

## Ingest Data (Build Vector Index)
This will:
- Read the bills dataset
- Chunk each bill into manageable pieces
- Generate embeddings using sentence-transformers
- Store everything in ./chroma_store

```bash
python ingest.py
```

You should see output like:
```text
Ingested 500+ chunks into Chroma.
```

## Run the API
Start FastAPI:
```bash
uvicorn app:app --reload
```
Then open:
http://127.0.0.1:8000/docs

## Test the API
Windows example:
```bash
curl -X POST "http://127.0.0.1:8000/ask" ^
 -H "Content-Type: application/json" ^
 -d "{\"question\":\"Can I get information on cable company tax revenue?\"}"
```

Windows example:
```json
{
  "answer": "The bill discusses tax treatment for cable companies...",
  "sources": [
    {
      "bill_id": "bill_9",
      "title": "Cable Television Tax Fairness Act",
      "chunk_index": 2,
      "score": 0.231
    }
  ]
}
```

## How RAG Works Here
- Documents are chunked and embedded
- Embeddings are stored in ChromaDB
- Your question gets embedded
- Top relevant chunks are retrieved
- LLM generates answer using only retrieved context
- API returns answer + source references

## Package Breakdown

| Package               | What it is                                   | Why you’re using it                                                   |
| --------------------- | -------------------------------------------- | --------------------------------------------------------------------- |
| **fastapi**           | A modern Python framework for building APIs  | Used to expose your RAG system as a REST API like `/ask`              |
| **uvicorn[standard]** | A lightning-fast ASGI web server             | Runs your FastAPI app locally or in production                        |
| **openai**            | Official OpenAI Python SDK                   | Used to call embedding models and LLM chat models                     |
| **chromadb**          | A lightweight local vector database          | Stores document embeddings + does similarity search for RAG           |
| **pydantic[dotenv]**  | Data validation + settings via `.env`        | Validates API request/response models and loads environment variables |
| **python-dotenv**     | Loads environment variables from `.env` file | Lets you store your API keys safely instead of hardcoding them        |
| **pandas**            | Data analytics library                       | Used to load & process the online CSV dataset (BillSum data)          |
