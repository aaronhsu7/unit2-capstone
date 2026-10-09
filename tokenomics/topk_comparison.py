# tokenomics/topk_comparison.py
"""Compare answers with top_k=5 (old) and top_k=3 (new) on the same questions.

Runs each document question through the qualitative agent with both settings,
using the same prompt, and records input/output tokens, whether the answer
contains each key fact, and the reviewer's verdict. This is the quality check
for the top_k change: the token saving only counts if answers stay complete.

Calls Gemini 4 times per question (2 answers + 2 reviews).

Usage:  python -m tokenomics.topk_comparison
"""
import time
from functools import partial
from google.genai import errors as genai_errors
from agents import qualitative
from validation.reviewer import review_answer

# question -> short phrases a complete answer should contain (case-insensitive).
# Looser than the exact source wording, because answers paraphrase.
CHECKS = {
    "How do we handle customer complaints?": [
        "1 business hour", "1 hour", "4 business hours", "outcome", "team lead",
        "10%", "root cause", "95%",
    ],
    "Explain the code review process": [
        "engineer", "400", "codeowners", "security champions", "1 business day",
        "squash", "senior engineer", "retrospective", "4 working hours",
    ],
    "What is our company's security policy?": [
        "four", "least privilege", "24 hours", "14 characters", "full-disk",
        "1 hour", "soc 2", "14 days", "6 months", "disciplinary",
    ],
}

def with_retries(fn, *args, attempts=8, wait=60):
    for i in range(attempts):
        try:
            return fn(*args)
        except genai_errors.APIError as e:
            print(f"    Gemini error {e.code}, retrying in {wait}s ({i + 1}/{attempts})")
            time.sleep(wait)
    raise RuntimeError("Gemini unavailable")

def run_with_top_k(question: str, top_k: int) -> dict:
    original = qualitative.retrieve
    qualitative.retrieve = partial(original, top_k=top_k)
    try:
        return with_retries(qualitative.run, question)
    finally:
        qualitative.retrieve = original

def main():
    rows = []
    for question, checks in CHECKS.items():
        for top_k in (5, 3):
            result = run_with_top_k(question, top_k)
            answer = result["answer"].lower()
            missing = [c for c in checks if c not in answer]
            review = with_retries(review_answer, question, result["answer"], result["chunks"])
            rows.append((question, top_k, result["input_tokens"], result["output_tokens"],
                         len(checks) - len(missing), len(checks), missing, review))
            print(f"\n{question}  [top_k={top_k}]")
            print(f"    tokens: {result['input_tokens']} in / {result['output_tokens']} out")
            print(f"    key facts in answer: {len(checks) - len(missing)}/{len(checks)}"
                  + (f"  missing: {missing}" if missing else ""))
            print(f"    reviewer: {review['warning'] or 'all claims supported'}")
            for issue in review["issues"]:
                print(f"      - [{issue.get('problem')}] {issue.get('claim')}")

    print("\nSummary")
    print(f"{'question':<42}{'top_k':>6}{'input':>8}{'output':>8}{'facts':>8}  reviewer")
    for q, k, i, o, f, t, _, r in rows:
        print(f"{q[:40]:<42}{k:>6}{i:>8}{o:>8}{f:>5}/{t:<3} {'ok' if r['supported'] else str(len(r['issues'])) + ' issue(s)'}")

if __name__ == "__main__":
    main()
