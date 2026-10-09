# check_history.py
"""Check that follow-up questions are rewritten into correct standalone questions.

Each case gives a short conversation history and a follow-up. The rewrite must
contain every word in `must_include` and none in `must_exclude`. Only the
rewrite step is tested (one Gemini call per case), not the full pipeline.

Usage:  python check_history.py
"""
from agents.manager import rewrite_followup

REVENUE_2025 = {
    "standalone": "What was total revenue by region in 2025?",
    "sql": "SELECT region, SUM(revenue) FROM sales WHERE strftime('%Y', date) = '2025' GROUP BY region",
    "answer": "North America had the highest revenue in 2025 ($6,511,620), followed by EMEA ($4,018,500), APAC ($3,810,540) and LATAM ($2,071,200).",
}
PARENTAL_LEAVE = {
    "standalone": "How many weeks of parental leave do employees get?",
    "sql": None,
    "answer": "Employees receive 16 weeks of fully paid parental leave [Source 1].",
}
SATISFACTION = {
    "standalone": "What is the average employee satisfaction score?",
    "sql": "SELECT ROUND(AVG(satisfaction_score), 2) FROM employees",
    "answer": "The average employee satisfaction score is 6.96 out of 10.",
}

# (name, history, follow-up, must_include, must_exclude)
CASES = [
    ("Change one filter ('what about')", [REVENUE_2025],
     "What about 2024?", ["2024", "region", "revenue"], ["2025"]),
    ("Same topic, new detail ('they')", [PARENTAL_LEAVE],
     "And how many wellbeing days do they get?", ["wellbeing", "employees"], []),
    ("Refers to a figure ('that')", [SATISFACTION],
     "Is that above the industry benchmark?", ["6.96", "benchmark"], []),
    ("New topic stays unchanged", [REVENUE_2025],
     "Explain the code review process", ["code review"], ["2025", "region", "revenue"]),
]

def main():
    passed = 0
    for name, history, followup, include, exclude in CASES:
        rewritten = rewrite_followup(followup, history)
        low = rewritten.lower()
        missing = [w for w in include if w.lower() not in low]
        unwanted = [w for w in exclude if w.lower() in low]
        ok = not missing and not unwanted
        passed += ok
        print(f"\n[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"    previous:  {history[-1]['standalone']}")
        print(f"    follow-up: {followup}")
        print(f"    rewritten: {rewritten}")
        if missing:
            print(f"    missing:   {missing}")
        if unwanted:
            print(f"    unwanted:  {unwanted}")
    print(f"\n{passed}/{len(CASES)} follow-ups rewritten correctly.")

if __name__ == "__main__":
    main()
