"""
ingest.py — Parse Chat Data.txt and index all knowledge entries into ChromaDB
Run this once before starting the chatbot:  python ingest.py
"""

import re
import os
import chromadb
from sentence_transformers import SentenceTransformer

KNOWLEDGE_FILE = "/Users/shishirraj/Documents/Chat Data.txt"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "tranzact_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 100


def parse_knowledge_entries(file_path: str) -> list[dict]:
    """Parse the Chat Data.txt file and extract all knowledge entries."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by the '------' separator lines
    sections = re.split(r"-{4,}", content)

    entries = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Match "Knowledge N" header at the start of the section
        match = re.match(r"Knowledge\s+(\d+)\s*\n(.*)", section, re.DOTALL)
        if match:
            knowledge_num = int(match.group(1))
            knowledge_content = match.group(2).strip()
            if knowledge_content and len(knowledge_content) > 20:
                entries.append(
                    {
                        "id": f"knowledge_{knowledge_num}",
                        "content": knowledge_content[:3000],  # cap per entry
                        "knowledge_num": knowledge_num,
                    }
                )

    return entries


def build_index(file_path: str = KNOWLEDGE_FILE, db_path: str = CHROMA_DB_PATH):
    print(f"\n{'='*50}")
    print("  TranZact Knowledge Base Ingestion")
    print(f"{'='*50}\n")

    # Parse
    print("Step 1/3  Parsing knowledge entries...")
    entries = parse_knowledge_entries(file_path)
    print(f"          Found {len(entries)} knowledge entries.\n")

    # Load embedding model
    print("Step 2/3  Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("          Model loaded.\n")

    # ChromaDB setup
    print("Step 3/3  Embedding & storing in ChromaDB...")
    client = chromadb.PersistentClient(path=db_path)

    # Drop existing collection to rebuild clean
    try:
        client.delete_collection(COLLECTION_NAME)
        print("          Dropped existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(entries)
    for i in range(0, total, BATCH_SIZE):
        batch = entries[i : i + BATCH_SIZE]
        texts = [e["content"] for e in batch]
        ids = [e["id"] for e in batch]
        metadatas = [{"knowledge_num": e["knowledge_num"]} for e in batch]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        processed = min(i + BATCH_SIZE, total)
        pct = int(processed / total * 100)
        bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
        print(f"          [{bar}] {processed}/{total} ({pct}%)", end="\r")

    print(f"\n\n  Done! {total} entries indexed in '{db_path}'.")
    print("  You can now run:  streamlit run app.py\n")


if __name__ == "__main__":
    build_index()
