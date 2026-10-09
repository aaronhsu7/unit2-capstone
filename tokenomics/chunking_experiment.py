# tokenomics/chunking_experiment.py
"""Compare chunk size / top_k settings for retrieval, without calling Gemini.

For each setting, the documents are chunked and embedded into a temporary
in-memory ChromaDB collection, and each test question is retrieved. Two things
are measured:

- Context size: words sent to Gemini as context (the qualitative agent's input
  tokens are almost all context; measured at ~1.2 tokens per word).
- Coverage: how many of the key facts a full answer needs are present in the
  retrieved context. If a fact isn't retrieved, the answer can't include it.

Usage:  python -m tokenomics.chunking_experiment
"""
import os
import chromadb
from sentence_transformers import SentenceTransformer
from ingest import chunk_document

DOCS_PATH = "./data/documents"

# (chunk_size words, overlap words, top_k)
SETTINGS = [
    (500, 50, 5),   # current
    (500, 50, 3),
    (300, 50, 4),
    (300, 50, 5),
    (250, 50, 5),
    (250, 50, 6),
    (200, 40, 6),
    (200, 40, 8),
]

# question -> phrases that must be in the context for a complete answer.
# Taken from the source documents; these are the facts checked by hand in
# README Trust-but-Verify Examples 4 and 6 and the security policy test.
FACTS = {
    "How do we handle customer complaints?": [
        "within 1 business hour", "First response within 1 hour",
        "First response within 4 business hours", "Never close a ticket without telling the customer the outcome",
        "escalated automatically to the Support Team Lead", "up to 10% of a customer's monthly fee",
        "root cause analysis", "95% of complaints",
    ],
    "Explain the code review process": [
        "at least one other engineer", "fewer than 400 changed lines", "CODEOWNERS",
        "Security Champions", "first response within 1 business day", "squash merging",
        "1 approval from any senior engineer", "retrospective review", "Median time to first review",
    ],
    "What is our company's security policy?": [
        "four classification levels", "least privilege", "within 24 hours", "at least 14 characters",
        "full-disk encryption", "within 1 hour of discovery", "SOC 2 Type II",
        "within 14 days of joining", "maximum of 6 months", "disciplinary action",
    ],
    "How many weeks of parental leave do employees get?": ["16 weeks of fully paid parental leave"],
    "What discount can an account executive approve?": ["Up to 10%: Account Executive may approve"],
    "What are the incident severity levels?": ["SEV1:", "SEV2:", "SEV3:", "SEV4:"],
}

def load_documents() -> dict:
    docs = {}
    for filename in sorted(os.listdir(DOCS_PATH)):
        if filename.endswith(".txt"):
            with open(os.path.join(DOCS_PATH, filename)) as f:
                docs[filename] = f.read()
    return docs

def evaluate(model, docs, chunk_size, overlap, top_k):
    client = chromadb.EphemeralClient()
    name = f"exp-{chunk_size}-{overlap}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name)
    for filename, text in docs.items():
        chunks = chunk_document(text, chunk_size, overlap)
        collection.add(documents=chunks, embeddings=model.encode(chunks).tolist(),
                       ids=[f"{filename}-{i}" for i in range(len(chunks))])

    results = {}
    for question, facts in FACTS.items():
        found = collection.query(query_embeddings=model.encode([question]).tolist(), n_results=top_k)
        context = " ".join(found["documents"][0])
        # Chunks are word-joined, so compare with normalised whitespace.
        norm = " ".join(context.split()).lower()
        covered = sum(" ".join(f.split()).lower() in norm for f in facts)
        results[question] = (covered, len(facts), len(context.split()))
    return results

def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = load_documents()
    print(f"{'chunk/overlap/top_k':<20}{'facts covered':>15}{'avg context words':>20}   per question (covered/total)")
    for chunk_size, overlap, top_k in SETTINGS:
        r = evaluate(model, docs, chunk_size, overlap, top_k)
        covered = sum(c for c, _, _ in r.values())
        total = sum(t for _, t, _ in r.values())
        avg_words = sum(w for _, _, w in r.values()) / len(r)
        per_q = "  ".join(f"{c}/{t}" for c, t, _ in r.values())
        label = f"{chunk_size}/{overlap}/{top_k}"
        print(f"{label:<20}{covered:>8}/{total:<6}{avg_words:>20.0f}   {per_q}")

if __name__ == "__main__":
    main()
