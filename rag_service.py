# # rag_service.py
# import os
# import re
# import uuid
# from typing import Optional, List

# import fitz  # PyMuPDF
# from fastapi import FastAPI, File, UploadFile, Query
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# import chromadb

# # --- config ---
# CHROMA_PATH = "./chroma_db"
# ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# # Init FastAPI app
# app = FastAPI(title="RAG Service")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=ALLOWED_ORIGINS,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Setup ChromaDB
# client = chromadb.PersistentClient(path=CHROMA_PATH)

# # ---------------------------
# # Helpers
# # ---------------------------
# def normalize_text(s: str) -> str:
#     return re.sub(r"\s+", " ", s).strip()

# def sanitize_collection_name(name: str) -> str:
#     sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
#     if len(sanitized) < 3:
#         sanitized = f"col_{sanitized}"
#     return sanitized

# def smart_chunk_document(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
#     """
#     Semantic chunking that preserves document structure.
#     Similar to how Claude processes documents - by logical sections.
#     """
#     chunks = []
    
#     # First, try to split by clear section breaks (double newlines)
#     sections = re.split(r'\n\s*\n+', text)
    
#     current_chunk = []
#     current_length = 0
    
#     for section in sections:
#         section = section.strip()
#         if not section:
#             continue
        
#         section_length = len(section)
        
#         # If this section alone is bigger than chunk_size, split it by sentences
#         if section_length > chunk_size:
#             # Split into sentences
#             sentences = re.split(r'(?<=[.!?])\s+', section)
            
#             for sentence in sentences:
#                 sentence = sentence.strip()
#                 if not sentence:
#                     continue
                
#                 sentence_length = len(sentence)
                
#                 if current_length + sentence_length > chunk_size and current_chunk:
#                     # Save current chunk
#                     chunk_text = " ".join(current_chunk)
#                     chunks.append(chunk_text)
                    
#                     # Keep last N chars for overlap
#                     overlap_text = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
#                     current_chunk = [overlap_text, sentence]
#                     current_length = len(overlap_text) + sentence_length
#                 else:
#                     current_chunk.append(sentence)
#                     current_length += sentence_length
#         else:
#             # Section fits, check if adding it exceeds chunk_size
#             if current_length + section_length > chunk_size and current_chunk:
#                 # Save current chunk
#                 chunk_text = " ".join(current_chunk)
#                 chunks.append(chunk_text)
                
#                 # Start new chunk with overlap
#                 overlap_text = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
#                 current_chunk = [overlap_text, section]
#                 current_length = len(overlap_text) + section_length
#             else:
#                 current_chunk.append(section)
#                 current_length += section_length
    
#     # Add final chunk
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))
    
#     # Clean and filter
#     final_chunks = []
#     for chunk in chunks:
#         chunk = chunk.strip()
#         # Only keep chunks that are substantial
#         if len(chunk) > 50:
#             final_chunks.append(chunk)
    
#     # If no chunks created, return whole text
#     if not final_chunks and text.strip():
#         final_chunks = [text.strip()]
    
#     return final_chunks

# # ---------------------------
# # Models
# # ---------------------------
# class QueryReq(BaseModel):
#     query: str
#     top_k: int = 8  # Increased to get more context

# # ---------------------------
# # Health
# # ---------------------------
# @app.get("/health")
# def health():
#     return {"ok": True}

# # ---------------------------
# # Upload file
# # ---------------------------
# @app.post("/upload_file")
# async def upload_file(file: UploadFile = File(...), chunk_size: int = Query(500)):
#     upload_id = str(uuid.uuid4())
#     safe_upload_id = sanitize_collection_name(upload_id)
#     ns_collection = client.get_or_create_collection(safe_upload_id)

#     file_bytes = await file.read()
#     text = ""

#     if file.filename.lower().endswith(".pdf"):
#         doc = fitz.open(stream=file_bytes, filetype="pdf")
#         for page in doc:
#             text += page.get_text("text") + "\n"
#     elif file.filename.lower().endswith(".txt"):
#         text = file_bytes.decode("utf-8", errors="ignore")
#         text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]+', ' ', text)

#     # ✅ USE UNIVERSAL SMART CHUNKING
#     chunks = smart_chunk_document(text, chunk_size=chunk_size, overlap=100)
    
#     # Debug: Print chunks to see what we're storing
#     print(f"\n=== CHUNKS FOR {file.filename} ===")
#     for i, chunk in enumerate(chunks[:5]):  # Print first 5 chunks
#         print(f"\nCHUNK {i}:")
#         print(chunk[:200])  # First 200 chars
#     print(f"Total chunks: {len(chunks)}\n")

#     if len(chunks) == 0:
#         chunks = [f"[GENERAL] {text}"]

#     added = 0
#     for i, chunk in enumerate(chunks):
#         emb = model.encode([chunk])[0].tolist()
#         cid = f"{safe_upload_id}_{i}"
#         ns_collection.add(
#             ids=[cid],
#             documents=[chunk],
#             embeddings=[emb],
#             metadatas=[{"namespace": safe_upload_id, "chunk_index": i}]
#         )
#         added += 1

#     print(f"✅ Uploaded file: {file.filename}, uploadId: {safe_upload_id}, chunks: {added}")
#     return {"status": "uploaded", "uploadId": safe_upload_id, "chunks": added}

# # ---------------------------
# # Query
# # ---------------------------
# @app.post("/query")
# def query_docs(req: QueryReq, namespace: Optional[str] = Query(None)):
#     if not namespace:
#         return {"chunks": []}
#     try:
#         safe_ns = sanitize_collection_name(namespace.strip())
#         ns_collection = client.get_or_create_collection(safe_ns)
        
#         total_docs = len(ns_collection.get()['ids'])
#         print(f"\n🔍 Querying: '{req.query}'")
#         print(f"Collection: {safe_ns}, Total docs: {total_docs}")

#         query_emb = model.encode([req.query])[0].tolist()
#         results = ns_collection.query(
#             query_embeddings=[query_emb],
#             n_results=min(req.top_k, total_docs)  # Don't request more than available
#         )

#         docs_list = results.get("documents", [[]])
#         docs_for_query = docs_list[0] if docs_list else []
        
#         # Debug: Print retrieved chunks
#         print(f"Retrieved {len(docs_for_query)} chunks:")
#         for i, chunk in enumerate(docs_for_query):
#             print(f"  Chunk {i}: {chunk[:100]}...")

#         cleaned_chunks = [normalize_text(chunk) for chunk in docs_for_query]
#         return {"chunks": cleaned_chunks}
#     except Exception as e:
#         print(f"❌ Query failed: {e}")
#         return {"chunks": [], "error": str(e)}

# # ---------------------------
# # Debug: View chunks for an upload
# # ---------------------------
# @app.get("/debug/{upload_id}")
# def debug_upload(upload_id: str):
#     try:
#         safe_upload_id = sanitize_collection_name(upload_id.strip())
#         ns_collection = client.get_or_create_collection(safe_upload_id)
        
#         all_data = ns_collection.get()
#         chunks = all_data.get('documents', [])
        
#         return {
#             "uploadId": safe_upload_id,
#             "totalChunks": len(chunks),
#             "chunks": [
#                 {
#                     "index": i,
#                     "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk,
#                     "length": len(chunk)
#                 }
#                 for i, chunk in enumerate(chunks)
#             ]
#         }
#     except Exception as e:
#         return {"error": str(e)}

# # ---------------------------
# # Delete one upload namespace
# # ---------------------------
# @app.delete("/delete_upload/{upload_id}")
# def delete_upload(upload_id: str):
#     try:
#         safe_upload_id = sanitize_collection_name(upload_id.strip())
#         client.delete_collection(safe_upload_id)
#         print(f"🗑️ Deleted collection: {safe_upload_id}")
#         return {"status": "deleted", "uploadId": safe_upload_id}
#     except Exception as e:
#         return {"status": "error", "error": str(e)}

# # ---------------------------
# # Admin: clear all
# # ---------------------------
# @app.delete("/clear_all")
# def clear_docs():
#     import shutil
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)
#     global client
#     client = chromadb.PersistentClient(path=CHROMA_PATH)
#     print("🧹 Cleared all ChromaDB data")
#     return {"status": "chroma_db cleared"}

# # ---------------------------
# # Run
# # ---------------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("rag_service:app", host="0.0.0.0", port=8000, reload=True)








########### new ###############
# rag_service.py
import os
import re
import uuid
from typing import Optional, List
import subprocess
import tempfile
import asyncio

import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import io

# --- config ---
CHROMA_PATH = "./chroma_db"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Voice service paths
WHISPER_MODEL_PATH = "/home/ec2-user/voice_service/whisper.cpp/models/ggml-base.en.bin"
WHISPER_CLI_PATH = "/home/ec2-user/voice_service/whisper.cpp/build/bin/whisper-cli"
TEMP_DIR = "/tmp/audio_files"

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

# Init FastAPI app
app = FastAPI(title="RAG + Voice Service")

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

def smart_chunk_document(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Semantic chunking that preserves document structure.
    Similar to how Claude processes documents - by logical sections.
    """
    chunks = []
    
    # First, try to split by clear section breaks (double newlines)
    sections = re.split(r'\n\s*\n+', text)
    
    current_chunk = []
    current_length = 0
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
        
        section_length = len(section)
        
        # If this section alone is bigger than chunk_size, split it by sentences
        if section_length > chunk_size:
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', section)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_length = len(sentence)
                
                if current_length + sentence_length > chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = " ".join(current_chunk)
                    chunks.append(chunk_text)
                    
                    # Keep last N chars for overlap
                    overlap_text = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
                    current_chunk = [overlap_text, sentence]
                    current_length = len(overlap_text) + sentence_length
                else:
                    current_chunk.append(sentence)
                    current_length += sentence_length
        else:
            # Section fits, check if adding it exceeds chunk_size
            if current_length + section_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_text = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text
                current_chunk = [overlap_text, section]
                current_length = len(overlap_text) + section_length
            else:
                current_chunk.append(section)
                current_length += section_length
    
    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Clean and filter
    final_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        # Only keep chunks that are substantial
        if len(chunk) > 50:
            final_chunks.append(chunk)
    
    # If no chunks created, return whole text
    if not final_chunks and text.strip():
        final_chunks = [text.strip()]
    
    return final_chunks

# ---------------------------
# Models
# ---------------------------
class QueryReq(BaseModel):
    query: str
    top_k: int = 8  # Increased to get more context

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "en-US-AriaNeural"  # Default voice

# ---------------------------
# Health
# ---------------------------
@app.get("/")
def root():
    return {
        "service": "RAG + Voice Service",
        "status": "running",
        "endpoints": {
            "rag": ["/upload_file", "/query", "/debug/{upload_id}", "/delete_upload/{upload_id}", "/clear_all"],
            "voice": ["/voice/transcribe", "/voice/synthesize", "/voice/health"]
        }
    }

@app.get("/health")
def health():
    return {"ok": True}

# =============================
# VOICE ENDPOINTS (NEW)
# =============================

@app.post("/voice/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Transcribe audio file using whisper.cpp
    Accepts: WAV, MP3, M4A, WEBM, OGG
    Returns: Transcribed text
    """
    temp_audio_path = None
    temp_wav_path = None
    
    try:
        print(f"📝 Transcribing audio: {audio.filename}")
        
        # Save uploaded audio with unique name
        file_ext = audio.filename.split('.')[-1] if '.' in audio.filename else 'webm'
        temp_audio_path = os.path.join(TEMP_DIR, f"input_{uuid.uuid4().hex}.{file_ext}")
        
        with open(temp_audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        print(f"✅ Saved audio to: {temp_audio_path}")
        
        # Convert to WAV if needed (whisper.cpp works best with WAV)
        temp_wav_path = os.path.join(TEMP_DIR, f"input_{uuid.uuid4().hex}.wav")
        
        # Convert to 16kHz mono WAV using ffmpeg
        convert_cmd = [
            "ffmpeg", "-i", temp_audio_path,
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",       # Mono
            "-c:a", "pcm_s16le",  # 16-bit PCM
            "-y",             # Overwrite
            temp_wav_path
        ]
        
        print(f"🔄 Converting audio to WAV...")
        result = subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
        print(f"✅ Conversion complete")
        
        # Run whisper transcription
        whisper_cmd = [
            WHISPER_CLI_PATH,
            "-m", WHISPER_MODEL_PATH,
            "-f", temp_wav_path,
            "-nt",  # No timestamps
            "-l", "en"  # English language
        ]
        
        print(f"🎤 Running Whisper transcription...")
        result = subprocess.run(
            whisper_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Extract transcription from output
        transcription = result.stdout.strip()
        
        # Clean up the output (remove whisper metadata)
        lines = transcription.split('\n')
        text_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines and metadata lines
            if line and not line.startswith('[') and not line.startswith('whisper_'):
                text_lines.append(line)
        
        final_text = ' '.join(text_lines).strip()
        
        print(f"✅ Transcription complete: '{final_text[:100]}...'")
        
        return {
            "success": True,
            "transcription": final_text,
            "language": "en"
        }
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else str(e)
        print(f"❌ Transcription failed: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {error_msg}"
        )
    except Exception as e:
        print(f"❌ Error processing audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )
    finally:
        # Cleanup temp files
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except:
                pass
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except:
                pass


@app.post("/voice/synthesize")
async def synthesize_speech(request: TTSRequest):
    """
    Convert text to speech using edge-tts (Microsoft Edge TTS)
    Returns: Audio file (MP3)
    """
    try:
        print(f"🔊 Synthesizing speech: '{request.text[:50]}...'")
        
        import edge_tts
        
        # Generate audio
        output_path = os.path.join(TEMP_DIR, f"output_{uuid.uuid4().hex}.mp3")
        
        # Use edge-tts to generate speech
        communicate = edge_tts.Communicate(
            text=request.text,
            voice=request.voice
        )
        
        await communicate.save(output_path)
        
        print(f"✅ TTS complete, reading audio file...")
        
        # Read the audio file
        with open(output_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)
        
        print(f"✅ Sending audio response ({len(audio_data)} bytes)")
        
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3"
            }
        )
        
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="edge-tts not installed. Run: pip install edge-tts"
        )
    except Exception as e:
        print(f"❌ TTS generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"TTS generation failed: {str(e)}"
        )


@app.get("/voice/health")
async def voice_health_check():
    """Check if whisper and TTS dependencies are available"""
    whisper_available = os.path.exists(WHISPER_CLI_PATH) and os.path.exists(WHISPER_MODEL_PATH)
    
    try:
        import edge_tts
        tts_available = True
    except ImportError:
        tts_available = False
    
    # Check ffmpeg
    ffmpeg_available = False
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        ffmpeg_available = True
    except:
        pass
    
    return {
        "whisper_available": whisper_available,
        "whisper_cli_path": WHISPER_CLI_PATH if whisper_available else "Not found",
        "whisper_model_path": WHISPER_MODEL_PATH if whisper_available else "Not found",
        "tts_available": tts_available,
        "ffmpeg_available": ffmpeg_available,
        "status": "ready" if (whisper_available and tts_available and ffmpeg_available) else "incomplete",
        "temp_dir": TEMP_DIR
    }


@app.get("/voice/voices")
async def list_voices():
    """List available TTS voices"""
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        
        # Filter English voices
        english_voices = [
            {
                "name": v["ShortName"],
                "gender": v["Gender"],
                "locale": v["Locale"]
            }
            for v in voices if v["Locale"].startswith("en-")
        ]
        
        return {
            "voices": english_voices[:20],  # Return first 20 English voices
            "total": len(english_voices)
        }
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="edge-tts not installed"
        )

# =============================
# EXISTING RAG ENDPOINTS (UNCHANGED)
# =============================

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

    # ✅ USE UNIVERSAL SMART CHUNKING
    chunks = smart_chunk_document(text, chunk_size=chunk_size, overlap=100)
    
    # Debug: Print chunks to see what we're storing
    print(f"\n=== CHUNKS FOR {file.filename} ===")
    for i, chunk in enumerate(chunks[:5]):  # Print first 5 chunks
        print(f"\nCHUNK {i}:")
        print(chunk[:200])  # First 200 chars
    print(f"Total chunks: {len(chunks)}\n")

    if len(chunks) == 0:
        chunks = [f"[GENERAL] {text}"]

    added = 0
    for i, chunk in enumerate(chunks):
        emb = model.encode([chunk])[0].tolist()
        cid = f"{safe_upload_id}_{i}"
        ns_collection.add(
            ids=[cid],
            documents=[chunk],
            embeddings=[emb],
            metadatas=[{"namespace": safe_upload_id, "chunk_index": i}]
        )
        added += 1

    print(f"✅ Uploaded file: {file.filename}, uploadId: {safe_upload_id}, chunks: {added}")
    return {"status": "uploaded", "uploadId": safe_upload_id, "chunks": added}


@app.post("/query")
def query_docs(req: QueryReq, namespace: Optional[str] = Query(None)):
    if not namespace:
        return {"chunks": []}
    try:
        safe_ns = sanitize_collection_name(namespace.strip())
        ns_collection = client.get_or_create_collection(safe_ns)
        
        total_docs = len(ns_collection.get()['ids'])
        print(f"\n🔍 Querying: '{req.query}'")
        print(f"Collection: {safe_ns}, Total docs: {total_docs}")

        query_emb = model.encode([req.query])[0].tolist()
        results = ns_collection.query(
            query_embeddings=[query_emb],
            n_results=min(req.top_k, total_docs)  # Don't request more than available
        )

        docs_list = results.get("documents", [[]])
        docs_for_query = docs_list[0] if docs_list else []
        
        # Debug: Print retrieved chunks
        print(f"Retrieved {len(docs_for_query)} chunks:")
        for i, chunk in enumerate(docs_for_query):
            print(f"  Chunk {i}: {chunk[:100]}...")

        cleaned_chunks = [normalize_text(chunk) for chunk in docs_for_query]
        return {"chunks": cleaned_chunks}
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return {"chunks": [], "error": str(e)}


@app.get("/debug/{upload_id}")
def debug_upload(upload_id: str):
    try:
        safe_upload_id = sanitize_collection_name(upload_id.strip())
        ns_collection = client.get_or_create_collection(safe_upload_id)
        
        all_data = ns_collection.get()
        chunks = all_data.get('documents', [])
        
        return {
            "uploadId": safe_upload_id,
            "totalChunks": len(chunks),
            "chunks": [
                {
                    "index": i,
                    "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    "length": len(chunk)
                }
                for i, chunk in enumerate(chunks)
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@app.delete("/delete_upload/{upload_id}")
def delete_upload(upload_id: str):
    try:
        safe_upload_id = sanitize_collection_name(upload_id.strip())
        client.delete_collection(safe_upload_id)
        print(f"🗑️ Deleted collection: {safe_upload_id}")
        return {"status": "deleted", "uploadId": safe_upload_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.delete("/clear_all")
def clear_docs():
    import shutil
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    global client
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    print("🧹 Cleared all ChromaDB data")
    return {"status": "chroma_db cleared"}

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag_service:app", host="0.0.0.0", port=8000, reload=True)