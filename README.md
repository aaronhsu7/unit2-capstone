# Unit 2 Capstone: Multi-Agent RAG System with Claude

## Project Overview

You will build a **Python CLI-based Retrieval-Augmented Generation (RAG) system** powered by Claude. The system uses specialised agents to answer both qualitative and quantitative questions about enterprise documentation.

The capstone covers the full pipeline: retrieval, prompt design, Claude API integration, and output validation. You must demonstrate trust-but-verify in practice by evaluating Claude's output, building guardrails, and accounting for tokenomics in your prompt design choices. A working application is not sufficient on its own. The validation layer and the tokenomics documentation are required deliverables.

---

## What You Will Build

A CLI application with three coordinated agents:

- **Manager Agent** — accepts user queries, classifies them as qualitative or quantitative, routes to the correct agent, and synthesises the final response
- **Qualitative Agent** — performs semantic search against a vector database, retrieves relevant document chunks, and uses Claude to generate a cited, grounded response
- **Quantitative Agent** — converts natural language questions to SQL, executes queries against a SQLite database, and uses Claude to interpret and present the results

All Claude interactions must pass through a validation layer before the response is returned to the user.

---

## Step 1: Review the System Architecture

```
User Query (CLI)
      │
      ▼
┌─────────────────┐
│  Manager Agent  │  ── classifies query ──► qualitative / quantitative / both
└─────────────────┘
      │                          │
      ▼                          ▼
┌──────────────────┐    ┌───────────────────┐
│ Qualitative Agent│    │ Quantitative Agent │
│                  │    │                   │
│ • Vector DB      │    │ • SQLite DB        │
│   (ChromaDB)     │    │ • NL → SQL         │
│ • Semantic search│    │ • Query execution  │
│ • Claude for     │    │ • Claude for       │
│   generation     │    │   interpretation   │
└──────────────────┘    └───────────────────┘
      │                          │
      └──────────┬───────────────┘
                 ▼
      ┌─────────────────────┐
      │  Validation Layer   │  ── checks grounding, flags issues
      └─────────────────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │  Tokenomics Logger  │  ── logs token usage and cost per query
      └─────────────────────┘
                 │
                 ▼
        Response to User
```

---

## Step 2: Setup

### Project structure

```
unit2-capstone/
├── agents/
│   ├── manager.py
│   ├── qualitative.py
│   └── quantitative.py
├── validation/
│   └── validator.py
├── tokenomics/
│   └── logger.py
├── data/
│   ├── documents/        # source documents for ingestion
│   └── database.sqlite   # SQLite database for quantitative queries
├── ingest.py             # document ingestion pipeline
├── main.py               # CLI entry point
├── .env                  # API keys and config — never committed
├── requirements.txt
└── README.md
```

### Environment setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install anthropic
pip install chromadb
pip install sentence-transformers
pip install python-dotenv

pip freeze > requirements.txt
```

### Environment variables

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_key_here
```

Add `.env` to `.gitignore` before your first commit. The API key must never appear in your codebase.

### Smoke test

Before building agents, confirm Claude is reachable:

```python
# smoke_test.py
import anthropic
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=50,
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print(message.content[0].text)
print(f"Input tokens: {message.usage.input_tokens}")
print(f"Output tokens: {message.usage.output_tokens}")
```

```bash
python smoke_test.py
```

Every learner must get a successful response before building further.

---

## Step 3: Build the Retrieval Pipeline

### Document ingestion

Write `ingest.py` to chunk your source documents and store embeddings in ChromaDB. Use the provided enterprise documentation set, or source your own — consulting reports, policy documents, or technical specifications work well.

```python
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
```

```bash
python ingest.py
```

Test retrieval before connecting Claude — confirm chunks are semantically relevant to your queries.

---

## Step 4: Build the Qualitative Agent

The qualitative agent retrieves relevant chunks from ChromaDB and passes them to Claude with a carefully designed prompt. Claude must be instructed to stay within the provided context.

```python
# agents/qualitative.py
import chromadb
import anthropic
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()
model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query: str, top_k: int = 5) -> list[dict]:
    chroma = chromadb.PersistentClient(path="./data/chroma")
    collection = chroma.get_collection("enterprise-docs")
    embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=embedding, n_results=top_k)
    return [
        {
            "content": doc,
            "source": meta["source"],
            "chunk": meta["chunk"]
        }
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

def build_prompt(query: str, chunks: list[dict]) -> str:
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Source {i+1}: {chunk['source']}]\n{chunk['content']}\n\n"

    return f"""You are a helpful enterprise documentation assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I cannot find this information in the provided documents."
Always cite the source number(s) you used.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

def run(query: str) -> dict:
    chunks = retrieve(query)
    prompt = build_prompt(query, chunks)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "answer": message.content[0].text,
        "chunks": chunks,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens
    }
```

> **Prompt design checkpoint:** Review every learner's `build_prompt` function before they wire it into the manager. A prompt that does not explicitly instruct Claude to stay within the provided context will hallucinate. The system prompt and the "cannot find" instruction are both non-negotiable.

---

## Step 5: Build the Quantitative Agent

The quantitative agent translates natural language to SQL using Claude, executes the query against SQLite, and interprets the results.

```python
# agents/quantitative.py
import sqlite3
import anthropic
from dotenv import load_dotenv
load_dotenv()

client = anthropic.Anthropic()

SCHEMA_CONTEXT = """
Available tables:
- sales(id, region, product, revenue, date, units_sold)
- customers(id, name, industry, churn_date, satisfaction_score)
- employees(id, department, satisfaction_score, tenure_years)
"""

def validate_sql(query: str) -> dict:
    blocked = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
    for word in blocked:
        if word in query.upper():
            return {"valid": False, "reason": f"Blocked keyword: {word}"}
    if not query.strip().upper().startswith("SELECT"):
        return {"valid": False, "reason": "Only SELECT queries are permitted"}
    return {"valid": True, "reason": "OK"}

def generate_sql(query: str) -> dict:
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"{SCHEMA_CONTEXT}\n\nGenerate a SQL query for: {query}\n\nReturn ONLY the SQL query, nothing else."
        }]
    )
    return {
        "sql": message.content[0].text.strip(),
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens
    }

def run(query: str) -> dict:
    sql_result = generate_sql(query)
    sql = sql_result["sql"]
    validation = validate_sql(sql)

    if not validation["valid"]:
        return {
            "answer": f"Query blocked: {validation['reason']}",
            "sql": sql,
            "rows": [],
            "validation": "FAILED",
            "input_tokens": sql_result["input_tokens"],
            "output_tokens": sql_result["output_tokens"]
        }

    try:
        conn = sqlite3.connect("./data/database.sqlite")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        conn.close()

        # Use Claude to interpret the results
        interpretation = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": f"The user asked: {query}\n\nSQL query used: {sql}\n\nResults:\nColumns: {cols}\nData: {rows[:20]}\n\nProvide a clear, concise interpretation of these results."
            }]
        )

        return {
            "answer": interpretation.content[0].text,
            "sql": sql,
            "columns": cols,
            "rows": rows,
            "validation": "PASSED",
            "input_tokens": sql_result["input_tokens"] + interpretation.usage.input_tokens,
            "output_tokens": sql_result["output_tokens"] + interpretation.usage.output_tokens
        }

    except Exception as e:
        return {
            "answer": f"Query execution failed: {str(e)}",
            "sql": sql,
            "rows": [],
            "validation": "ERROR",
            "input_tokens": sql_result["input_tokens"],
            "output_tokens": sql_result["output_tokens"]
        }
```

> **Trust-but-verify checkpoint:** The `validate_sql` function is non-negotiable. No query should execute against the database without passing through the validator first. This is not optional — it is the practical application of trust-but-verify in a data context.

---

## Step 6: Build the Validation Layer

Every response from Claude — regardless of which agent produced it — passes through the validator before the user sees it.

```python
# validation/validator.py

def validate_qualitative(answer: str, chunks: list[dict]) -> dict:
    sources_cited = []
    for i, chunk in enumerate(chunks):
        if f"Source {i+1}" in answer:
            sources_cited.append(chunk["source"])

    grounded = len(sources_cited) > 0
    refused = "cannot find" in answer.lower()

    return {
        "is_grounded": grounded,
        "refused_to_answer": refused,
        "sources_cited": sources_cited,
        "flag": not grounded and not refused,
        "warning": "Response may not be grounded in source documents" if (not grounded and not refused) else None
    }

def validate_quantitative(answer: str, sql: str, validation_status: str) -> dict:
    return {
        "sql_validated": validation_status == "PASSED",
        "sql_blocked": validation_status == "FAILED",
        "execution_error": validation_status == "ERROR",
        "flag": validation_status != "PASSED",
        "warning": f"SQL validation status: {validation_status}" if validation_status != "PASSED" else None
    }
```

> **Validation checkpoint:** Test your validator by deliberately giving Claude irrelevant context (qualitative) or submitting a non-SELECT query (quantitative). Both should be flagged. The validation layer must catch failures before the user sees them.

---

## Step 7: Build the Tokenomics Logger

Log token usage for every query. This is a required deliverable — not optional.

```python
# tokenomics/logger.py
import json
from datetime import datetime

COST_PER_1K_INPUT  = 0.003   # Update to current Claude pricing
COST_PER_1K_OUTPUT = 0.015

def log(query: str, agent: str, input_tokens: int, output_tokens: int):
    input_cost  = (input_tokens  / 1000) * COST_PER_1K_INPUT
    output_cost = (output_tokens / 1000) * COST_PER_1K_OUTPUT
    total_cost  = input_cost + output_cost

    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "agent": agent,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(total_cost, 6),
        "cost_per_1000_queries": round(total_cost * 1000, 2)
    }

    print(f"\n[TOKENOMICS] Agent: {agent} | Input: {input_tokens} | Output: {output_tokens} | Cost: ${total_cost:.6f}")

    with open("tokenomics_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry
```

---

## Step 8: Build the Manager Agent

```python
# agents/manager.py
import anthropic
from dotenv import load_dotenv
from agents import qualitative, quantitative
from validation.validator import validate_qualitative, validate_quantitative
from tokenomics.logger import log
load_dotenv()

client = anthropic.Anthropic()

def classify(query: str) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": f"""Classify this query as exactly one of: qualitative, quantitative, both.

qualitative = questions about policies, processes, procedures, explanations, documentation
quantitative = questions about numbers, metrics, trends, comparisons, SQL-queryable data
both = questions that need both document search and data analysis

Query: {query}

Reply with one word only: qualitative, quantitative, or both."""
        }]
    )
    route = message.content[0].text.strip().lower()
    log(query, "manager-classifier", message.usage.input_tokens, message.usage.output_tokens)
    return route if route in ["qualitative", "quantitative", "both"] else "qualitative"

def run(query: str):
    print(f"\nQuery: {query}")
    route = classify(query)
    print(f"Route: {route}")

    qual_result = None
    quant_result = None

    if route in ["qualitative", "both"]:
        qual_result = qualitative.run(query)
        validation = validate_qualitative(qual_result["answer"], qual_result["chunks"])
        log(query, "qualitative", qual_result["input_tokens"], qual_result["output_tokens"])
        if validation["flag"]:
            print(f"\n⚠️  VALIDATION WARNING: {validation['warning']}")
        print(f"\n[Qualitative]\n{qual_result['answer']}")

    if route in ["quantitative", "both"]:
        quant_result = quantitative.run(query)
        validation = validate_quantitative(
            quant_result["answer"],
            quant_result["sql"],
            quant_result["validation"]
        )
        log(query, "quantitative", quant_result["input_tokens"], quant_result["output_tokens"])
        if validation["flag"]:
            print(f"\n⚠️  VALIDATION WARNING: {validation['warning']}")
        print(f"\n[Quantitative]\n{quant_result['answer']}")
        print(f"SQL used: {quant_result['sql']}")
```

---

## Step 9: Build the CLI Entry Point

```python
# main.py
from agents.manager import run

def main():
    print("Spoonful Enterprise RAG System")
    print("Type 'exit' to quit.\n")
    while True:
        query = input("Ask a question: ").strip()
        if query.lower() in ["exit", "quit"]:
            break
        if not query:
            continue
        run(query)

if __name__ == "__main__":
    main()
```

```bash
python main.py
```

---

## Step 10: Supported Query Types

Test all three query types before submission.

**Qualitative (document search):**
- "What is our company's security policy?"
- "Explain the code review process"
- "How do we handle customer complaints?"

**Quantitative (SQL + data):**
- "Show me monthly revenue trends"
- "What is our customer churn rate?"
- "Compare Q4 performance across regions"

**Complex multi-agent:**
- "How does our employee satisfaction compare to industry standards and what policies might impact this?"
- "Analyse our sales performance and recommend policy changes based on our customer success strategies"

---

## Trust-but-Verify Documentation

Your README must include a `## Trust-but-Verify` section that documents the following for at least three queries you ran during testing:

1. **The query you submitted**
2. **What Claude returned**
3. **What the validation layer flagged, if anything**
4. **What you accepted, what you changed, and why**
5. **One case where Claude produced an output you did not immediately trust** — and how you resolved it

This section is assessed. A description of what the validation layer does is not sufficient — it must contain real examples from your testing with specific outputs and specific decisions you made.

---

## Must-Have Checklist

> 🥉 Bronze — complete all must-haves

- [ ] Python 3.11 codebase with virtual environment and `requirements.txt`
- [ ] Document ingestion pipeline (`ingest.py`) that chunks and embeds documents into ChromaDB
- [ ] Manager Agent that classifies queries and routes to the correct agent(s)
- [ ] Qualitative Agent that retrieves from ChromaDB and calls Claude with a grounded prompt
- [ ] Quantitative Agent with SQLite integration, NL-to-SQL via Claude, and `validate_sql` blocking non-SELECT queries before execution
- [ ] Validation layer that runs on every Claude response before it reaches the user
- [ ] Tokenomics logger that records input tokens, output tokens, and cost per query
- [ ] Claude (`claude-sonnet-4-6`) used as the model throughout — no other LLM providers
- [ ] API key stored in `.env` — never hardcoded or committed
- [ ] CLI entry point (`main.py`) that supports all three query types
- [ ] `## Trust-but-Verify` section in README with real examples from testing
- [ ] `tokenomics_log.jsonl` included in submission showing queries run during testing

---

## Stretch Goals

> 🥈 Silver — Bronze plus one of these

- [ ] Conversation history — the manager agent maintains context across a session so Claude can answer follow-up questions that reference prior queries
- [ ] A second validation strategy for the qualitative agent — run a second Claude call that acts as a reviewer, assessing whether the answer is supported by the retrieved context

> 🥇 Gold — Silver plus one of these

- [ ] Unit tests for each agent (manager, qualitative, quantitative, validator)
- [ ] Integration tests for multi-agent workflows
- [ ] Tokenomics optimisation — add a section to your README analysing your `tokenomics_log.jsonl` and identifying one concrete change you made to reduce token usage without degrading output quality

---

## Deliverables

- Python codebase meeting all requirements above
- Working CLI that handles all supported query types
- `tokenomics_log.jsonl` showing at least 10 queries run during testing
- `README.md` with:
  - Architecture diagram
  - Setup and usage instructions
  - `## Trust-but-Verify` section with real examples

---

## Tips for Success

- **Build and test each agent independently** before wiring them together. Run `ingest.py` and confirm retrieval works before touching Claude. Get the SQL agent returning correct queries for simple questions before handling complex ones.
- **Test the validation layer before the manager.** Feed Claude irrelevant context on purpose and confirm the validator flags it. This is not a nice-to-have — it is the point of the exercise.
- **Review your tokenomics log after every test run.** If a single query is costing significantly more than others, look at the prompt — long, unstructured context is almost always the cause.
- **Comment your prompts.** Every design decision in a prompt should have a comment explaining why it is there. Future you — and your reviewers — will thank you.
- **Ask for help if blocked.** Instructors are available. Do not spend more than 30 minutes stuck on the same issue without reaching out.

---

## References

- [Anthropic API documentation](https://docs.anthropic.com)
- [ChromaDB documentation](https://docs.trychroma.com)
- [Sentence Transformers documentation](https://www.sbert.net)
- [SQLite Python documentation](https://docs.python.org/3/library/sqlite3.html)
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
