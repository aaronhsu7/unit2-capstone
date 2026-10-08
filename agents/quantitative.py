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