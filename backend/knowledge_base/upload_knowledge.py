"""
FraudX Analyst - Knowledge Base Uploader (Full Version)
=========================================================
Processes PDF documents + built-in knowledge articles,
chunks them, generates embeddings, and uploads to Pinecone.

Run ONCE to populate Pinecone:
    python upload_knowledge.py
"""

import os, sys, time, uuid, re
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

import PyPDF2
from pinecone import Pinecone
from google import genai

import numpy as np
from google.genai import types

from knowledge_content import KNOWLEDGE_BASE

# ── Config ─────────────────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX_NAME", "fraudx-knowledge")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
#EMBEDDING_MODEL  = "text-embedding-004"
DOCS_DIR         = os.path.join(os.path.dirname(__file__), "documents")
CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 100

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ── PDF extraction ─────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: str) -> str:
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            total  = len(reader.pages)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        print(f"         → {total} pages, {len(text):,} chars extracted")
    except Exception as e:
        print(f"  ⚠️ Failed to extract: {e}")
    return text


# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    return text.strip()


# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text   = text.strip()
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for delimiter in ['. ', '.\n', '! ', '? ']:
                pos = text.rfind(delimiter, start + chunk_size // 2, end)
                if pos > 0:
                    end = pos + 2
                    break
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start = end - overlap
    return chunks


# ── Embedding ──────────────────────────────────────────────────────────────────
def get_embedding(text: str, retries: int = 3) -> list:
    for attempt in range(retries):
        try:
            response = gemini_client.models.embed_content(
            model    = "models/gemini-embedding-001",
            contents = text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT",output_dimensionality=2048,)
            )
            
            vec = np.array(response.embeddings[0].values, dtype=np.float32)
            vec = vec / np.linalg.norm(vec)
            
            return vec.tolist()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = (attempt + 1) * 15
                print(f"         ⏳ Rate limited — waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"         ⚠️ Embedding error: {e}")
                break
    return []


# ── Upload to Pinecone ─────────────────────────────────────────────────────────
def upload_vectors(index, vectors: list):
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"  📤 Uploaded batch {i//batch_size + 1} ({len(batch)} vectors)")


# ── Process PDFs ───────────────────────────────────────────────────────────────
def process_pdfs() -> list:
    if not os.path.exists(DOCS_DIR):
        print(f"  ⚠️ Documents folder not found: {DOCS_DIR}")
        return []

    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print("  ⚠️ No PDF files found in documents/")
        return []

    print(f"\n  📚 Processing {len(pdf_files)} PDF documents …\n")
    all_vectors = []

    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(DOCS_DIR, filename)
        title    = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        print(f"  [{i+1}/{len(pdf_files)}] {title[:70]}")

        raw_text = extract_pdf_text(pdf_path)
        clean    = clean_text(raw_text)
        if not clean:
            print(f"         ⚠️ No text extracted — skipping")
            continue

        chunks = chunk_text(clean)
        print(f"         → {len(chunks)} chunks to embed")

        doc_vectors = 0
        for j, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if not embedding:
                continue
            all_vectors.append({
                "id"      : f"pdf_{i}_{j}_{str(uuid.uuid4())[:6]}",
                "values"  : embedding,
                "metadata": {
                    "text"    : chunk,
                    "title"   : title[:100],
                    "category": "research",
                    "source"  : filename,
                    "chunk_id": j,
                }
            })
            doc_vectors += 1
            time.sleep(0.6)

        print(f"         ✅ {doc_vectors} vectors ready")

    return all_vectors


# ── Process built-in articles ──────────────────────────────────────────────────
def process_articles() -> list:
    print(f"\n  📝 Processing {len(KNOWLEDGE_BASE)} built-in articles …\n")
    all_vectors = []

    for i, entry in enumerate(KNOWLEDGE_BASE):
        print(f"  [{i+1}/{len(KNOWLEDGE_BASE)}] {entry['title']}")
        chunks = chunk_text(entry["content"])
        print(f"         → {len(chunks)} chunks")

        for j, chunk in enumerate(chunks):
            embedding = get_embedding(chunk)
            if not embedding:
                continue
            all_vectors.append({
                "id"      : f"article_{i}_{j}_{str(uuid.uuid4())[:6]}",
                "values"  : embedding,
                "metadata": {
                    "text"    : chunk,
                    "title"   : entry["title"],
                    "category": entry["category"],
                    "source"  : "FraudX Knowledge Base",
                    "chunk_id": j,
                }
            })
            time.sleep(0.6)

    return all_vectors


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  FraudX Analyst — Knowledge Base Uploader")
    print("=" * 60)

    print("\n  Connecting to Pinecone …")
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    stats = index.describe_index_stats()
    print(f"  ✅ Connected → {PINECONE_INDEX}")
    print(f"  Current vectors: {stats['total_vector_count']}")

    all_vectors  = []
    all_vectors += process_pdfs()
    all_vectors += process_articles()

    print(f"\n  Uploading {len(all_vectors)} total vectors to Pinecone …")
    upload_vectors(index, all_vectors)

    stats = index.describe_index_stats()
    print(f"\n{'=' * 60}")
    print(f"  ✅ Upload complete!")
    print(f"  Total uploaded  : {len(all_vectors)}")
    print(f"  Pinecone total  : {stats['total_vector_count']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
