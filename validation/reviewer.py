# validation/reviewer.py
"""Second validation strategy for the qualitative agent: an LLM reviewer.

validate_qualitative() only checks that "[Source N]" citations are present. It
passed answers with a wrong citation and a made-up figure, because it cannot
read what the cited source actually says (README Trust-but-Verify, Example 4).
This module sends the answer and the sources it cites to a second Gemini call,
which checks each claim against those sources.
"""
import json
import re
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from dotenv import load_dotenv
from config import MODEL
from tokenomics.logger import log, token_counts
load_dotenv()

client = genai.Client()  # reads GEMINI_API_KEY from .env

PROBLEM_TYPES = ["unsupported", "wrong_citation", "contradicted", "uncited"]

def cited_source_numbers(answer: str, n_chunks: int) -> list[int]:
    """Source numbers the answer cites, e.g. [1, 2] for "[Source 1, Source 2]"."""
    found = {int(n) for n in re.findall(r"Source\s+(\d+)", answer)}
    return sorted(n for n in found if 1 <= n <= n_chunks)

def build_review_prompt(answer: str, chunks: list[dict], cited: list[int]) -> str:
    # Prompt design decisions:
    # - Only the CITED chunks are sent, with their original Source numbers, so
    #   the reviewer can match each citation to the right text. In testing,
    #   40-60% of retrieved chunks were never used by the answer; sending them
    #   would add input tokens without helping the check.
    # - "Do not use outside knowledge": the question is whether the SOURCES
    #   support the answer, not whether the answer is true in general.
    # - Four named problem types, so issues are specific and countable:
    #   unsupported (cited source doesn't say it), wrong_citation (a different
    #   source shown says it), contradicted (a figure, deadline or name
    #   differs), uncited (factual claim with no citation).
    # - "Do not report correctly supported claims" and "wording differences
    #   that keep the meaning": keeps the output short (output is the
    #   expensive part) and avoids noise that would train users to ignore it.
    # - JSON output (response_mime_type) so the result is parsed, not guessed.
    sources = "".join(
        f"[Source {n}: {chunks[n - 1]['source']}]\n{chunks[n - 1]['content']}\n\n" for n in cited
    )
    return f"""You are reviewing an answer written by another assistant. Check it strictly against the source excerpts below. Do not use outside knowledge.

SOURCES (only the sources the answer cites are shown):
{sources or "(the answer cites no sources)"}
ANSWER TO REVIEW:
{answer}

Check every factual claim in the answer against the source(s) it cites. Report a problem only when:
- unsupported: the cited source does not state the claim, and no other source shown states it
- wrong_citation: the claim is stated in a different source shown above, not the one cited
- contradicted: a figure, deadline, name or condition differs from what the source says
- uncited: the claim is factual but has no citation
Do not report claims that are correctly supported. Do not report wording differences that keep the meaning.

Reply with JSON only, in this form:
{{"supported": true or false, "issues": [{{"claim": "...", "cited": "Source N", "problem": "unsupported|wrong_citation|contradicted|uncited", "explanation": "..."}}]}}"""

def review_answer(query: str, answer: str, chunks: list[dict]) -> dict:
    """Ask a second Gemini call whether the answer is supported by its cited sources.

    Returns {"reviewed": bool, "supported": bool, "issues": [...], "warning": str|None}.
    """
    cited = cited_source_numbers(answer, len(chunks))
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=build_review_prompt(answer, chunks, cited),
            # temperature=0 so the same answer gets the same review.
            config=types.GenerateContentConfig(max_output_tokens=1024, temperature=0,
                                               response_mime_type="application/json")
        )
        log(query, "qualitative-reviewer", *token_counts(response))
        result = json.loads(response.text or "")
    except (genai_errors.APIError, json.JSONDecodeError) as e:
        # A missing review must be visible, never silently treated as "passed".
        reason = f"Gemini error {e.code}" if isinstance(e, genai_errors.APIError) else "unreadable reply"
        return {"reviewed": False, "supported": False, "issues": [],
                "warning": f"Reviewer could not check this answer ({reason}); it has not been reviewed"}

    issues = [i for i in result.get("issues", []) if i.get("problem") in PROBLEM_TYPES]
    supported = bool(result.get("supported")) and not issues
    warning = None
    if not supported:
        n = len(issues)
        warning = f"Reviewer found {n} claim{'s' if n != 1 else ''} not supported by the cited sources"
    return {"reviewed": True, "supported": supported, "issues": issues, "warning": warning}

def format_issues(issues: list[dict]) -> str:
    lines = []
    for i in issues:
        lines.append(f"    - [{i.get('problem')}] {i.get('claim')} (cited: {i.get('cited', 'none')})")
        if i.get("explanation"):
            lines.append(f"      {i['explanation']}")
    return "\n".join(lines)
