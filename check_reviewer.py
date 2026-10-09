# check_reviewer.py
"""Check the reviewer (second validation strategy) against answers with known errors.

Each case is a hand-written answer about real retrieved chunks, with a known
correct verdict. Unlike validate_qualitative(), which only checks that
citations exist, the reviewer should catch wrong citations, changed figures
and invented facts. One Gemini call per case.

Usage:  python check_reviewer.py
"""
from agents.qualitative import retrieve
from validation.reviewer import review_answer, format_issues

COMPLAINTS = "How do we handle customer complaints?"
PARENTAL = "How many weeks of parental leave do employees get?"

# (name, question used for retrieval, answer, should the reviewer flag it?)
CASES = [
    ("Correct and correctly cited (should pass)", COMPLAINTS,
     "Every complaint must be logged as a ticket within 1 business hour [Source 1]. "
     "Credits above 25% and all refunds require approval from the Director of Customer Support and Finance [Source 2].",
     False),
    ("Wrong citation from README Example 4", COMPLAINTS,
     "Complaints from at-risk customers are always handled at priority P2 or higher [Source 1]. "
     "Credits above 25% require approval from the Director of Customer Support and Finance [Source 2].",
     True),
    ("Changed deadline (P1 is 1 hour, not 4)", COMPLAINTS,
     "P1 complaints must receive a first response within 4 hours [Source 1].",
     True),
    ("Invented rule with a citation", COMPLAINTS,
     "Customers who complain three times in a year receive a free month of service [Source 2].",
     True),
    ("Factual claim with no citation", COMPLAINTS,
     "Every complaint must be logged as a ticket within 1 business hour.",
     True),
    ("Made-up figure (the long-standing offline gap)", PARENTAL,
     "Spoonful offers 52 weeks of fully paid parental leave [Source 1].",
     True),
]

def main():
    passed = 0
    for name, question, answer, should_flag in CASES:
        chunks = retrieve(question)
        review = review_answer(question, answer, chunks)
        flagged = not review["supported"]
        ok = review["reviewed"] and flagged == should_flag
        passed += ok
        print(f"\n[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"    answer:   {answer}")
        print(f"    expected: {'flag' if should_flag else 'pass'}, got: "
              f"{'not reviewed' if not review['reviewed'] else ('flag' if flagged else 'pass')}")
        if review["warning"]:
            print(f"    warning:  {review['warning']}")
        if review["issues"]:
            print(format_issues(review["issues"]))
    print(f"\n{passed}/{len(CASES)} reviewer verdicts correct.")

if __name__ == "__main__":
    main()
