"""
Ingestion pipeline: reads all markdown files in ../data, chunks them,
embeds the chunks, and stores them in a persistent Chroma collection
along with BM25-ready raw text for hybrid search.

Run: python src/ingest.py
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(__file__))

from chunking import chunk_document
from embeddings import get_embedder
import chromadb

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
BM25_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db", "bm25_corpus.json")
COLLECTION_NAME = "farm_advisor_kb"


def titleize(filename: str) -> str:
    name = filename.replace(".md", "").replace("_", " ")
    return name.title()


def main():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    md_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {DATA_DIR}")

    all_chunks = []
    for filename in md_files:
        title = titleize(filename)
        filepath = os.path.join(DATA_DIR, filename)
        chunks = chunk_document(filepath, title)
        all_chunks.extend(chunks)
        logger.info("Chunked %-40s -> %d chunks", filename, len(chunks))

    logger.info("Total chunks: %d", len(all_chunks))

    embedder = get_embedder()
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts)

    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)

    # Fresh collection each ingest run, keeps this idempotent for demos.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["chunk_id"] for c in all_chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"source": c["source"], "section": c["section"]} for c in all_chunks],
    )

    # Save raw corpus separately for BM25 sparse retrieval (hybrid search).
    with open(BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f)

    # Persist which embedding backend was actually used, so rag_engine.py
    # can load a matching query-time embedder.
    with open(os.path.join(DB_DIR, "embedder_backend.json"), "w") as f:
        json.dump({"backend": embedder.backend}, f)

    logger.info("Ingest complete. Backend used: %s", embedder.backend)
    logger.info("Vector store persisted at: %s", DB_DIR)


if __name__ == "__main__":
    main()
