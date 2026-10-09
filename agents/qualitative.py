# agents/qualitative.py
import chromadb
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from config import MODEL
from tokenomics.logger import log, token_counts
load_dotenv()

client = genai.Client()  # reads GEMINI_API_KEY from .env
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
    # Each chunk is labelled "[Source N: filename]". N gives the model a short,
    # exact label to cite, and validate_qualitative() maps "Source N" back to
    # chunks[N-1] to decide whether the answer is grounded.
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"[Source {i+1}: {chunk['source']}]\n{chunk['content']}\n\n"

    # Prompt design decisions:
    # - "ONLY the context provided" + "do not guess": keeps answers traceable to
    #   Spoonful's documents instead of the model's general knowledge. Retrieval
    #   always returns top_k chunks, even for questions no document answers (e.g.
    #   the cryptocurrency test), so this is what stops the model answering from
    #   unrelated chunks.
    # - Exact refusal sentence, "and nothing else": validate_qualitative()
    #   detects a refusal by the phrase "cannot find". When this instruction was
    #   removed (check_validation.py test 1b), Gemini declined in its own words
    #   and the refusal was not recognised. "Nothing else" stops a refusal
    #   followed by a guess ("I cannot find it, but it's probably...").
    #   Do not reword the sentence without updating the validator.
    # - Conditions, deadlines and consequences: answers to process and policy
    #   questions kept the main rules but dropped what was attached to them,
    #   e.g. the 2-day retrospective review required after a hotfix, or the
    #   disciplinary consequences in the security policy (README Example 4).
    #   This rule asks for them explicitly. Trade-off: longer answers (593 -> 920
    #   output tokens on the complaints question), and output is the most
    #   expensive part of this call.
    # - Partial answers: if the context covers only part of the question, the
    #   model answers that part (with citations) and says what is missing,
    #   rather than refusing outright or filling the gap with a guess.
    # - Citation format "[Source N]": the validator looks for "Source N" to
    #   decide an answer is grounded, so the exact format is specified. It only
    #   checks that a citation is present, not that the source supports the
    #   claim (README Trust-but-Verify, Example 2).
    # - Question after the context, ending with "ANSWER:": the question is the
    #   last thing the model reads, and the reply starts with the answer itself.
    # - Token cost: the context is almost all of the input. 5 chunks of up to
    #   500 words is roughly 3,000+ input tokens per question (estimate, to be
    #   confirmed from tokenomics_log.jsonl). top_k and the chunk size in
    #   ingest.py are the main levers for reducing it. The instructions
    #   themselves are under 150 tokens.
    return f"""You are a helpful enterprise documentation assistant.
Answer the question using ONLY the context provided below. Do not guess or add information from general knowledge.
If the context does not contain the answer, reply with exactly this sentence and nothing else: "I cannot find this information in the provided documents."
If the context answers only part of the question, answer that part and state which part the documents do not cover.
When describing a rule or process, include the conditions, exceptions, deadlines and consequences attached to it in the context.
Cite the source of every fact in the form [Source N], using the source numbers shown in the context.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

def run(query: str) -> dict:
    chunks = retrieve(query)
    prompt = build_prompt(query, chunks)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        # 2048 caps the cost of a single reply (output is ~8x the price of
        # input). It was 1024, but after the "conditions, deadlines and
        # consequences" rule was added, the code review answer reached 1,020
        # tokens and was cut off mid-sentence, just before the hotfix deadline
        # (README Example 6). Typical answers use 600-1,000 tokens, so 2048
        # leaves headroom without raising the cost of normal answers.
        config=types.GenerateContentConfig(max_output_tokens=2048)
    )
    input_tokens, output_tokens = token_counts(response)
    log(query, "qualitative", input_tokens, output_tokens)
    # Gemini reports why it stopped writing. MAX_TOKENS means the answer hit the
    # output limit and is incomplete; the validator flags it, because a cut-off
    # answer still has citations and would otherwise pass as grounded.
    finish_reason = response.candidates[0].finish_reason if response.candidates else None
    return {
        "answer": response.text or "",
        "truncated": finish_reason == types.FinishReason.MAX_TOKENS,
        "chunks": chunks,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }
