import os
import uuid
import pandas as pd
import chromadb
from chromadb import Settings
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


#Load env
load_dotenv()

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str):
    return embed_model.encode(text).tolist()

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def main():
    df = pd.read_csv("bill_sum_data.csv")
    
    df.head(200)
    
    chroma_client = chromadb.PersistentClient(path="./chroma_store")
    
    collection = chroma_client.get_or_create_collection(name="bills")
    
    ids = []
    documents = []
    metadatas = []
    embeddings = []
    
    for idx, row in df.iterrows():
        text = str(row["text"])
        title = str(row.get("title", f"Bill {idx}"))
        bill_id = f"bill_{idx}"
        
        chunks = chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            chunk_id = f"{bill_id}_{i}_{uuid.uuid4().hex[:8]}"
            emb = get_embedding(chunk)
            
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
              "bill_id": bill_id,
              "title": title,
              "chunk_index": i  
            })
            embeddings.append(emb)
    
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"Ingested {len(ids)} chunks into Chroma.")
     
if __name__ == "__main__":
    main()  
        