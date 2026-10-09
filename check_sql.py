# check_sql.py
"""Check that the quantitative agent writes correct SQL for simple questions.

For each question there is a hand-written reference query. The script asks Gemini
for SQL, runs it through validate_sql, executes both queries, and compares the
results. Column names and row order are ignored; numbers are compared to 2 d.p.

This calls Gemini once per question (SQL generation only, no interpretation step).

Usage:  python check_sql.py
"""
import sqlite3
from agents.quantitative import generate_sql, validate_sql

DB_PATH = "./data/database.sqlite"

# question -> reference SQL that gives the correct answer
TESTS = {
    "How many employees are there?":
        "SELECT COUNT(*) FROM employees",
    "What is the total revenue across all sales?":
        "SELECT SUM(revenue) FROM sales",
    "How many customers have churned?":
        "SELECT COUNT(*) FROM customers WHERE churn_date IS NOT NULL",
    "What is the average employee satisfaction score for each department?":
        "SELECT department, AVG(satisfaction_score) FROM employees GROUP BY department",
    "What was the total revenue for each region in 2025?":
        "SELECT region, SUM(revenue) FROM sales WHERE date LIKE '2025-%' GROUP BY region",
    "How many customers are there in each industry?":
        "SELECT industry, COUNT(*) FROM customers GROUP BY industry",
}

def normalise(rows):
    """Make results comparable: round floats, ignore row order."""
    def cell(v):
        return round(v, 2) if isinstance(v, float) else v
    return sorted(tuple(cell(v) for v in row) for row in rows)

def run_sql(conn, sql):
    return conn.execute(sql).fetchall()

def main():
    conn = sqlite3.connect(DB_PATH)
    passed = 0

    for question, reference in TESTS.items():
        result = generate_sql(question)
        sql = result["sql"]
        validation = validate_sql(sql)

        print(f"\n{question}")
        print(f"    generated: {sql!r}")

        if not validation["valid"]:
            print(f"    [BLOCKED] {validation['reason']}")
            continue

        try:
            got = run_sql(conn, sql)
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue

        expected = run_sql(conn, reference)
        if normalise(got) == normalise(expected):
            print(f"    [PASS] {got[:5]}")
            passed += 1
        else:
            print(f"    [MISMATCH]")
            print(f"      got:      {got[:5]}")
            print(f"      expected: {expected[:5]}")

    conn.close()
    print(f"\n{passed}/{len(TESTS)} questions returned the correct result.")

if __name__ == "__main__":
    main()
