# rag_service.py
import os
import re
import uuid
from typing import Optional, List

import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb

# --- config ---
CHROMA_PATH = "./chroma_db"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Init FastAPI app
app = FastAPI(title="RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Setup ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)

# ---------------------------
# Helpers
# ---------------------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def sanitize_collection_name(name: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    if len(sanitized) < 3:
        sanitized = f"col_{sanitized}"
    return sanitized

# ---------------------------
# Models
# ---------------------------
class QueryReq(BaseModel):
    query: str
    top_k: int = 3

# ---------------------------
# Health
# ---------------------------
@app.get("/health")
def health():
    return {"ok": True}

# ---------------------------
# Upload file
# ---------------------------
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...), chunk_size: int = Query(500)):
    upload_id = str(uuid.uuid4())
    safe_upload_id = sanitize_collection_name(upload_id)
    ns_collection = client.get_or_create_collection(safe_upload_id)

    file_bytes = await file.read()
    text = ""

    if file.filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text("text") + "\n"
    elif file.filename.lower().endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]+', ' ', text)
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = re.sub(r' +', ' ', text).strip()
        if len(text) == 0:
            text = " "

    min_chunk_length = 5 if file.filename.lower().endswith(".txt") else 30
    raw_chunks: List[str] = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    chunks = [c.strip() for c in raw_chunks if len(c.strip()) >= min_chunk_length]

    if len(chunks) == 0:
        chunks = [text]

    added = 0
    for i, chunk in enumerate(chunks):
        emb = model.encode([chunk])[0].tolist()
        cid = f"{safe_upload_id}_{i}"
        ns_collection.add(
            ids=[cid],
            documents=[chunk],
            embeddings=[emb],
            metadatas=[{"namespace": safe_upload_id}]
        )
        added += 1

    print(f"Uploaded file: {file.filename}, uploadId: {safe_upload_id}, chunks: {added}")
    return {"status": "uploaded", "uploadId": safe_upload_id, "chunks": added}

# ---------------------------
# Query
# ---------------------------
@app.post("/query")
def query_docs(req: QueryReq, namespace: Optional[str] = Query(None)):
    if not namespace:
        return {"chunks": []}
    try:
        safe_ns = sanitize_collection_name(namespace.strip())
        ns_collection = client.get_or_create_collection(safe_ns)
        print(f"Querying collection: {safe_ns}, total documents: {len(ns_collection.get()['ids'])}")

        query_emb = model.encode([req.query])[0].tolist()
        results = ns_collection.query(
            query_embeddings=[query_emb],
            n_results=req.top_k
        )
        print(f"Raw query results: {results}")

        docs_list = results.get("documents", [[]])
        docs_for_query = docs_list[0] if docs_list else []

        cleaned_chunks = [normalize_text(chunk) for chunk in docs_for_query]
        return {"chunks": cleaned_chunks}
    except Exception as e:
        print(f"Query failed: {e}")
        return {"chunks": [], "error": str(e)}

# ---------------------------
# Delete one upload namespace
# ---------------------------
@app.delete("/delete_upload/{upload_id}")
def delete_upload(upload_id: str):
    try:
        safe_upload_id = sanitize_collection_name(upload_id.strip())
        client.delete_collection(safe_upload_id)
        return {"status": "deleted", "uploadId": safe_upload_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ---------------------------
# Admin: clear all
# ---------------------------
@app.delete("/clear_all")
def clear_docs():
    import shutil
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    global client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return {"status": "chroma_db cleared"}

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_service:app", host="0.0.0.0", port=8000, reload=True)
