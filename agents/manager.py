# agents/manager.py
import json
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from dotenv import load_dotenv
from config import MODEL
from agents import qualitative, quantitative
from validation.validator import validate_qualitative, validate_quantitative, validate_synthesis
from tokenomics.logger import log, token_counts
load_dotenv()

client = genai.Client()  # reads GEMINI_API_KEY from .env

def classify(query: str) -> str:
    # Prompt design decisions:
    # - Three fixed labels with a one-line definition each: the definitions use
    #   the same kinds of words users ask with ("policies", "trends",
    #   "comparisons"), so the model can match questions to a route.
    # - Quantitative is defined by the DATABASE, not by "numbers": the first
    #   version said "questions about numbers", and "How many weeks of parental
    #   leave do employees get?" was routed to SQL, although the answer is in a
    #   policy and the database has no leave data (tested 2026-10-08). The
    #   definitions now name the tables, and say that numbers stated in
    #   policies (allowances, deadlines, limits) are qualitative.
    # - "both" exists for questions like "how does satisfaction compare to
    #   industry standards", which need the database AND the documents.
    # - "One lowercase word, no punctuation": the reply is parsed directly, so
    #   anything other than the bare label would not match. Short output also
    #   keeps this call cheap, since it runs on every query. The parsing below
    #   also strips punctuation, so "Quantitative." still routes correctly.
    # - Unrecognised replies fall back to "qualitative", the safer route: its
    #   prompt refuses when the documents have no answer, while a wrong SQL
    #   route would run a query. The fallback is printed, not hidden, so a
    #   misroute is visible to the user and in testing.
    response = client.models.generate_content(
        model=MODEL,
        contents=f"""Classify this query as exactly one of: qualitative, quantitative, both.

qualitative = questions about policies, processes, procedures, rules and documentation, including numbers stated in policies (e.g. leave allowances, response times, approval limits, targets)
quantitative = questions answered by querying company data: sales (revenue, units, regions, products, dates), customers (industry, churn, satisfaction scores) and employees (department, satisfaction scores, tenure)
both = questions that need both company data and policy documents

Query: {query}

Reply with one lowercase word and no punctuation: qualitative, quantitative, or both.""",
        # The answer is one word, but Gemini's thinking tokens count toward this
        # limit; a very small cap (e.g. 10) can return an empty answer.
        config=types.GenerateContentConfig(max_output_tokens=256)
    )
    raw = response.text or ""
    route = raw.strip().strip(".!\"'`").lower()
    log(query, "manager-classifier", *token_counts(response))
    if route not in ["qualitative", "quantitative", "both"]:
        print(f"\n⚠️  Classifier returned {raw!r}; defaulting to qualitative.")
        return "qualitative"
    return route

def print_rows(cols: list, rows: list, limit: int = 50):
    """Print query results as a plain table.

    The results are shown directly so the interpretation prompt doesn't have
    to repeat them (README Example 5), and so the user can check the
    interpretation's numbers against the data.
    """
    if not rows:
        return
    def fmt(v):
        return f"{v:,.2f}" if isinstance(v, float) else str(v)
    table = [[str(c) for c in cols]] + [[fmt(v) for v in row] for row in rows[:limit]]
    widths = [max(len(r[i]) for r in table) for i in range(len(cols))]
    print(f"\nResults ({len(rows)} rows):")
    for i, r in enumerate(table):
        print("  " + "  ".join(v.rjust(w) for v, w in zip(r, widths)))
        if i == 0:
            print("  " + "  ".join("-" * w for w in widths))
    if len(rows) > limit:
        print(f"  ... {len(rows) - limit} more rows not shown")

def split_question(query: str) -> dict:
    """Split a "both" question into a data part and a document part."""
    # Prompt design decisions:
    # - Why split at all: both agents used to receive the full question. For
    #   "How does our employee satisfaction compare to industry standards...",
    #   the SQL agent tried to answer the "industry standards" part from the
    #   database, compared employee satisfaction with CUSTOMER satisfaction,
    #   and grouped by department AND tenure (227 rows). The document agent
    #   meanwhile ignored the thresholds that matter for the data (README
    #   Example 7). Each agent should only get the part it can answer.
    # - Benchmarks, targets and thresholds are named as document content, so
    #   "compare to industry standards" goes to the documents, not to SQL.
    # - "At the level needed": asks for overall and per-group figures, so the
    #   data part returns numbers that can be compared with thresholds.
    # - JSON output (response_mime_type) so the reply can be parsed directly.
    #   If parsing fails, both agents get the original question, as before.
    prompt = f"""Split this question into two standalone questions for two different agents.

data_question: the part answered by querying the company database (tables: sales, customers, employees). Ask only for figures that are in the database, at the level needed to answer (for example overall and per department, region or year).
document_question: the part answered by company policy documents, including any benchmarks, targets, thresholds, rules or policies.

Question: {query}

Reply with JSON only: {{"data_question": "...", "document_question": "..."}}"""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=512,
                                               response_mime_type="application/json")
        )
        log(query, "manager-split", *token_counts(response))
        parts = json.loads(response.text or "")
        if parts.get("data_question") and parts.get("document_question"):
            return parts
    except (genai_errors.APIError, json.JSONDecodeError, AttributeError):
        pass
    print("\n⚠️  Could not split the question; sending the full question to both agents.")
    return {"data_question": query, "document_question": query}

def synthesize(query: str, parts: dict, qual_result: dict, quant_result: dict) -> dict:
    """Combine both agents' results into one answer."""
    rows = quant_result["rows"]
    shown = rows[:quantitative.MAX_ROWS_FOR_INTERPRETATION]
    row_note = f"all {len(rows)} rows" if len(rows) == len(shown) else f"first {len(shown)} of {len(rows)} rows"
    # Prompt design decisions:
    # - The synthesis is the only step that sees both results, so it is where
    #   the actual comparison happens ("is 6.96 above or below the 7.1
    #   benchmark?"). The instruction to compare and say whether each target or
    #   threshold is met is the core of this prompt.
    # - The data agent's raw rows are passed, not an interpretation, so the
    #   figures come straight from SQLite and one Gemini call is saved.
    # - Status lines tell the model when a side failed or was blocked, so it
    #   says which part couldn't be answered instead of filling the gap.
    # - [Source N] kept on document facts and [Data] on database figures:
    #   validate_synthesis() checks both markers, so every claim in the
    #   combined answer stays traceable to where it came from.
    # - Same grounding rule as the other prompts: no outside facts or causes.
    # - max_output_tokens=2048, and truncation is detected (README Example 6).
    prompt = f"""The user asked: {query}

Two agents each answered part of this question.

DOCUMENT AGENT
Question it answered: {parts["document_question"]}
Answer:
{qual_result["answer"] or "(no answer)"}

DATA AGENT
Question it answered: {parts["data_question"]}
Status: {quant_result["validation"]}
SQL used: {quant_result["sql"]}
Results ({row_note}):
Columns: {quant_result.get("columns", [])}
Data: {shown}

Write one combined answer to the user's question.
Rules:
- Use only the information above. Do not add outside facts, benchmarks, causes or assumptions.
- Connect the two parts: compare figures from the data with any benchmarks, targets or thresholds in the documents, and state whether each one is met.
- Keep the [Source N] citation on every fact taken from the document agent's answer. Put [Data] after every figure taken from the data agent's results.
- Quote figures exactly as given.
- If either agent had no answer or its status is not PASSED, say which part of the question could not be answered.
- Be concise."""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=2048)
    )
    log(query, "manager-synthesis", *token_counts(response))
    finish_reason = response.candidates[0].finish_reason if response.candidates else None
    return {"answer": response.text or "",
            "truncated": finish_reason == types.FinishReason.MAX_TOKENS}

def run_both(query: str):
    parts = split_question(query)
    print(f"Data question:     {parts['data_question']}")
    print(f"Document question: {parts['document_question']}")

    qual_result = qualitative.run(parts["document_question"])
    quant_result = quantitative.run(parts["data_question"], interpret=False)

    # Each agent's own checks still run, and their warnings are shown, so a
    # problem in one half isn't hidden by a fluent combined answer.
    qual_check = validate_qualitative(qual_result["answer"], qual_result["chunks"],
                                      qual_result.get("truncated", False))
    quant_check = validate_quantitative(quant_result["answer"], quant_result["sql"],
                                        quant_result["validation"])
    for label, check in [("Document agent", qual_check), ("Data agent", quant_check)]:
        if check["flag"]:
            print(f"\n⚠️  VALIDATION WARNING ({label}): {check['warning']}")

    try:
        combined = synthesize(query, parts, qual_result, quant_result)
    except genai_errors.APIError as e:
        # Synthesis failed: fall back to showing both parts separately.
        print(f"\n⚠️  Gemini was unavailable to combine the answers ({e.code}); showing both parts separately.")
        print(f"\n[Qualitative]\n{qual_result['answer']}")
        print(f"\nSQL used: {quant_result['sql']}")
        print_rows(quant_result.get("columns", []), quant_result["rows"])
        return

    check = validate_synthesis(
        combined["answer"], qual_result["chunks"],
        docs_answered=not qual_check["refused_to_answer"] and bool(qual_result["answer"]),
        data_answered=quant_result["validation"] == "PASSED" and bool(quant_result["rows"]),
        truncated=combined["truncated"])
    if check["flag"]:
        print(f"\n⚠️  VALIDATION WARNING (combined answer): {check['warning']}")

    print(f"\n[Combined answer]\n{combined['answer']}")
    # Map each Source number to its file, so citations can be checked by hand.
    print("\nSources:")
    for i, chunk in enumerate(qual_result["chunks"], start=1):
        print(f"  [Source {i}] {chunk['source']} (chunk {chunk['chunk']})")
    print(f"\nSQL used: {quant_result['sql']}")
    print_rows(quant_result.get("columns", []), quant_result["rows"])

def run(query: str):
    print(f"\nQuery: {query}")
    route = classify(query)
    print(f"Route: {route}")

    if route == "both":
        run_both(query)
        return

    if route == "qualitative":
        qual_result = qualitative.run(query)
        validation = validate_qualitative(qual_result["answer"], qual_result["chunks"],
                                          qual_result.get("truncated", False))
        if validation["flag"]:
            print(f"\n⚠️  VALIDATION WARNING: {validation['warning']}")
        print(f"\n[Qualitative]\n{qual_result['answer']}")

    if route == "quantitative":
        quant_result = quantitative.run(query)
        validation = validate_quantitative(
            quant_result["answer"],
            quant_result["sql"],
            quant_result["validation"]
        )
        if validation["flag"]:
            print(f"\n⚠️  VALIDATION WARNING: {validation['warning']}")
        print(f"\n[Quantitative]\n{quant_result['answer']}")
        print(f"SQL used: {quant_result['sql']}")
        print_rows(quant_result.get("columns", []), quant_result["rows"])
