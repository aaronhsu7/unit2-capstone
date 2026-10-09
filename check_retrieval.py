# check_retrieval.py
"""Check that ChromaDB search returns the right documents, without calling Gemini.

Each test question names the document that should come back first. Run after
ingest.py, and again whenever you change the documents or chunk settings.

Usage:  python check_retrieval.py
"""
from agents.qualitative import retrieve

TOP_K = 5  # same as the qualitative agent's default

# question -> file the top result should come from (None = answer not in any document)
TESTS = {
    "What is our company's security policy?": "information_security_policy.txt",
    "Explain the code review process": "code_review_process.txt",
    "How do we handle customer complaints?": "customer_complaint_handling.txt",
    "What is the industry benchmark for employee satisfaction?": "employee_engagement_and_wellbeing_policy.txt",
    "What is our target customer churn rate?": "customer_success_strategy.txt",
    "What discount can an account executive approve?": "sales_operations_policy.txt",
    "What are the incident severity levels?": "incident_response_runbook.txt",
    "What is the company's stance on cryptocurrency investments?": None,
}

def main():
    passed = 0
    checked = 0
    for question, expected in TESTS.items():
        hits = retrieve(question, top_k=TOP_K)
        top = hits[0]["source"]

        if expected is None:
            status = "INFO"  # nothing should match; check the hits look unrelated
        else:
            checked += 1
            if top == expected:
                status = "PASS"
                passed += 1
            elif expected in [h["source"] for h in hits]:
                status = "WEAK"  # right document retrieved, but not ranked first
            else:
                status = "FAIL"

        print(f"\n[{status}] {question}")
        if expected:
            print(f"    expected: {expected}")
        for rank, hit in enumerate(hits, start=1):
            preview = hit["content"][:70].replace("\n", " ")
            print(f"    {rank}. {hit['source']} #{hit['chunk']}: {preview}...")

    print(f"\n{passed}/{checked} questions returned the expected document first.")

if __name__ == "__main__":
    main()
