import os
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="./chroma_store")

collection = chroma_client.get_or_create_collection(name="bills")

app = FastAPI(title="RAG over bills")

class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    
class Source(BaseModel):
    bill_id: str
    title: str
    chunk_index: int
    score: float
    
class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    
def get_embedding(text: str):
    return embed_model.encode(text).tolist()
    
def build_prompt(question: str, docs: List[str], metadatas: List[dict]) -> str:
    context_blocks = []
    for doc, meta in zip(docs, metadatas):
        context_blocks.append(
            f"[{meta.get('title', '')}, chunk {meta.get('chunk_index', '')}]\n{doc}"
        )
    context_text = '\n\n---\n\n'.join(context_blocks)
    
    prompt = f"""
        You are a helpful assistant answering questions about US legislative bills.

        Use ONLY the information in the CONTEXT below. If the answer is not clearly
        contained in the context, say "I don't know based on the provided documents."

        CONTEXT:
        {context_text}

        QUESTION:
        {question}

        Answer in clear, concise English.
        """
    return prompt.strip()

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q_emb = get_embedding(req.question)
    
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=req.top_k
    )
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    scores = results["distances"][0]
    
    prompt = build_prompt(req.question, docs, metas)
    
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            { "role": "system", "content": "You are a knowledge assistant for US bills."},
            { "role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    answer = completion.choices[0].message.content
    
    source_objs = []
    
    for meta, score in zip(metas, scores):
        source_objs.append(
            Source(
                bill_id=meta.get("bill_id", ""),
                title=meta.get("title", ""),
                chunk_index=int(meta.get("chunk_index", -1)),
                score=float(score)
            )
        )
    
    return AskResponse(
        answer=answer,
        sources=source_objs
    )
    