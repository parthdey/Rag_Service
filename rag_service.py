from fastapi import FastAPI
# import chromadb  # Comment out temporarily
# import fitz      # Comment out temporarily
# from sentence_transformers import SentenceTransformer  # Comment out

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "RAG service is running"}

@app.get("/")
def root():
    return {"message": "RAG Service API"}

# Comment out RAG-specific endpoints temporarily