# ingest.py
import chromadb
from sentence_transformers import SentenceTransformer
import os

def chunk_document(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def ingest(docs_path: str):
    client = chromadb.PersistentClient(path="./data/chroma")
    collection = client.get_or_create_collection("enterprise-docs")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    for filename in os.listdir(docs_path):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(docs_path, filename)) as f:
            text = f.read()
        chunks = chunk_document(text)
        embeddings = model.encode(chunks).tolist()
        ids = [f"{filename}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
        print(f"Ingested {len(chunks)} chunks from {filename}")

if __name__ == "__main__":
    ingest("./data/documents")