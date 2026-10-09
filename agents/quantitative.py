# agents/quantitative.py
import re
import sqlite3
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from dotenv import load_dotenv
from config import MODEL
from tokenomics.logger import log, token_counts
load_dotenv()

client = genai.Client()  # reads GEMINI_API_KEY from .env

# Schema sent with every SQL request.
# Design decisions:
# - Column formats and meanings: the first version listed only table and column
#   names (~80 input tokens), so the model had to guess that dates are
#   'YYYY-MM-DD' text, that NULL churn_date means active, and what scale scores
#   use. Guessing wrong produces SQL that runs but returns the wrong answer,
#   which no validator catches.
# - Exact category values: WHERE clauses must match stored text exactly
#   (e.g. 'EMEA', not 'Europe'), so the allowed values are listed.
# - "SQLite": date functions differ between databases (strftime here, not
#   EXTRACT or YEAR()), so the dialect is named.
# - Quarter definitions: "Q4" is ambiguous without them; these match the fiscal
#   calendar in sales_operations_policy.txt.
# - Token cost: this raised the SQL prompt from ~80 to 383 input tokens
#   (measured), about 5x. At $0.30 per 1M input tokens that is ~$0.00009 more
#   per query, a negligible cost for fewer wrong answers.
SCHEMA_CONTEXT = """
SQLite database. Use SQLite syntax (e.g. strftime for dates).

Tables:
- sales(id, region, product, revenue, date, units_sold)
    region: 'North America', 'EMEA', 'APAC', 'LATAM'
    product: 'Spoonful Starter', 'Spoonful Pro', 'Spoonful Enterprise', 'Data Connect'
    revenue: net annual contract value in USD, after discount
    date: deal close date as 'YYYY-MM-DD' text, from 2024-01-01 to 2025-12-31
- customers(id, name, industry, churn_date, satisfaction_score)
    industry: 'Retail', 'Education', 'Technology', 'Logistics', 'Manufacturing', 'Finance', 'Healthcare'
    churn_date: date the customer cancelled, as 'YYYY-MM-DD' text; NULL if the customer is still active
    satisfaction_score: 1 to 10
- employees(id, department, satisfaction_score, tenure_years)
    department: 'Engineering', 'Sales', 'Customer Support', 'Customer Success', 'Product', 'Marketing', 'Finance', 'People', 'Operations'
    satisfaction_score: 1 to 10, from the latest engagement survey

Quarters are calendar quarters: Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, Q4 = Oct-Dec.
"""

# Maximum result rows sent to the interpretation prompt. 50 covers the largest
# common results (24 months of trends, 4 regions x 8 quarters = 32) while
# keeping input tokens bounded for unexpectedly large results.
MAX_ROWS_FOR_INTERPRETATION = 50

# Keywords that change the database or reach outside it.
BLOCKED_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
                    "CREATE", "ATTACH", "DETACH", "PRAGMA"]

def validate_sql(query: str) -> dict:
    # Design decisions (README Trust-but-Verify, Example 3):
    # - String literals are blanked out before checking, so a value like
    #   '%Dropship%' is not mistaken for the DROP keyword.
    # - Keywords are matched as whole words, so a column alias like
    #   last_updated no longer triggers UPDATE. The first version used a plain
    #   substring check and blocked safe queries like these.
    # - WITH ... SELECT (a read-only common table expression) is allowed
    #   alongside plain SELECT; any write inside it is still caught above.
    # - Only one statement: anything after a ";" is blocked.
    # - The query must read FROM a table. Gemini once answered a policy question
    #   with SELECT 'I am sorry...' when it was misrouted; a literal is not data.
    # This is the first layer of protection; run() also opens the database
    # read-only, so a write that slipped past this check would still fail.
    code = re.sub(r"'(?:[^']|'')*'", "''", query).strip().rstrip(";").strip()
    upper = code.upper()
    for word in BLOCKED_KEYWORDS:
        if re.search(rf"\b{word}\b", upper):
            return {"valid": False, "reason": f"Blocked keyword: {word}"}
    if ";" in code:
        return {"valid": False, "reason": "Only one statement is permitted"}
    if not re.match(r"(SELECT|WITH)\b", upper):
        return {"valid": False, "reason": "Only SELECT queries are permitted"}
    if not re.search(r"\bFROM\b", upper):
        return {"valid": False, "reason": "Query does not read from any table"}
    return {"valid": True, "reason": "OK"}

def strip_code_fences(text: str) -> str:
    """Remove a markdown code fence (```sql ... ```) wrapped around the whole reply.

    Gemini often fences its SQL even when asked not to. Only a fence around the
    entire reply is removed: SQL surrounded by extra prose is left as-is so
    validate_sql still rejects it.
    """
    match = re.fullmatch(r"```(?:sql)?\s*(.*?)\s*```", text.strip(), re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else text.strip()

def generate_sql(query: str) -> dict:
    # Prompt design decisions:
    # - Schema first, then the question: the model sees which tables, columns
    #   and values exist before reading what to compute.
    # - "No markdown or code fences": the reply is executed, so any extra text
    #   breaks it. Gemini wrapped SQL in ```sql fences even when told "ONLY the
    #   SQL" (README Trust-but-Verify, Example 1), so fences are now named
    #   explicitly. strip_code_fences() still handles them in code, because the
    #   prompt alone was not reliable.
    # - "A single SELECT statement": asks for read-only SQL up front, so normal
    #   questions don't produce SQL that validate_sql() would block. This is NOT
    #   the safety control: when asked to delete data or drop a table, Gemini
    #   wrote the destructive SQL anyway (Example 3). validate_sql() in run()
    #   is what protects the database.
    # - "Use only the tables, columns and values listed": stops the model
    #   inventing columns (e.g. a "quarter" column) or category spellings.
    # - max_output_tokens=256: generated SQL is short (11-18 output tokens
    #   measured on simple questions); the cap leaves room for longer queries
    #   while bounding cost if the model rambles.
    response = client.models.generate_content(
        model=MODEL,
        contents=(
            f"{SCHEMA_CONTEXT}\n"
            f"Write one SQL query that answers: {query}\n\n"
            "Rules:\n"
            "- Return only the SQL as plain text, with no markdown, code fences or explanation.\n"
            "- Use a single SELECT statement.\n"
            "- Use only the tables, columns and values listed above."
        ),
        config=types.GenerateContentConfig(max_output_tokens=256)
    )
    input_tokens, output_tokens = token_counts(response)
    log(query, "quantitative-sql", input_tokens, output_tokens)
    return {
        "sql": strip_code_fences(response.text or ""),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    }

def run(query: str, interpret: bool = True) -> dict:
    # interpret=False skips the Gemini interpretation call and returns only the
    # SQL results. The manager uses this for "both" questions, where its
    # synthesis step reads the rows directly; interpreting them here as well
    # would pay for a second explanation of the same data.
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

    # SQL execution and the Gemini interpretation call are handled separately.
    # The first version wrapped both in one try/except, so a Gemini outage after
    # the SQL had already succeeded was reported as "Query execution failed"
    # (seen during a parental leave question that was misrouted to SQL).
    try:
        # Read-only connection: a second layer of protection if a write ever
        # gets past validate_sql().
        conn = sqlite3.connect("file:./data/database.sqlite?mode=ro", uri=True)
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
        finally:
            conn.close()
    except sqlite3.Error as e:
        return {
            "answer": f"Query execution failed: {e}",
            "sql": sql,
            "rows": [],
            "validation": "ERROR",
            "input_tokens": sql_result["input_tokens"],
            "output_tokens": sql_result["output_tokens"]
        }

    if not interpret:
        return {
            "answer": "",
            "sql": sql,
            "columns": cols,
            "rows": rows,
            "validation": "PASSED",
            "input_tokens": sql_result["input_tokens"],
            "output_tokens": sql_result["output_tokens"]
        }

    # Use Gemini to interpret the results.
    # Prompt design decisions:
    # - Includes the original question, so the explanation answers what was
    #   asked rather than just describing the table.
    # - Includes the SQL, so the model can say how the numbers were computed
    #   (e.g. "average across all departments") and the user can check it.
    # - Columns listed separately from the rows, so values can be matched
    #   to their meaning.
    # - Row count stated explicitly: the first version sent rows[:20]
    #   without saying so, so with more results (e.g. 24 months of revenue)
    #   the model would describe the first 20 as the whole result. Now the
    #   limit is MAX_ROWS_FOR_INTERPRETATION and, if results are cut off,
    #   the prompt says how many rows exist and tells the model to say so.
    # - "Use only the numbers in these results": the first version had no
    #   grounding rule, so the model could add benchmarks, causes or
    #   advice that are not in the data. Interpretation stays factual;
    #   policy context comes from the qualitative agent instead.
    # - "Describe the patterns... do not list every row": with only the rule
    #   above, "Show me monthly revenue trends" got all 24 values copied back
    #   and no trend at all; 469 output tokens, 85% of the query's cost
    #   (README Example 5). The grounding rule was read as "don't interpret".
    #   The prompt now asks for patterns found IN the data (highest/lowest,
    #   change over time, seasonal peaks), while still banning outside facts
    #   and causes. The rows are printed as a table by the manager, so the
    #   model doesn't need to repeat them.
    # - Empty results: without this rule, the model may explain what the
    #   answer "would" be instead of saying no data matched.
    # - Known gap: the validation layer still does not check the
    #   interpretation's numbers against the rows. The rows and SQL are
    #   shown to the user alongside it so they can be checked by hand.
    # - max_output_tokens=512: "clear, concise" answers fit well within this.
    shown = rows[:MAX_ROWS_FOR_INTERPRETATION]
    if len(rows) > len(shown):
        row_note = f"showing the first {len(shown)} of {len(rows)} rows"
    else:
        row_note = f"all {len(rows)} rows"
    try:
        interpretation = client.models.generate_content(
            model=MODEL,
            contents=(
                f"The user asked: {query}\n\n"
                f"SQL query used: {sql}\n\n"
                f"Results ({row_note}):\n"
                f"Columns: {cols}\n"
                f"Data: {shown}\n\n"
                "Give a clear, concise answer to the user's question based on these results.\n"
                "Rules:\n"
                "- Describe the patterns in the data: for example the highest and lowest values, how values change over time, and any repeating or seasonal peaks. Quote only the figures needed to support each point.\n"
                "- Do not list every row. The full results are shown to the user separately.\n"
                "- Use only the numbers in these results. Do not add facts, benchmarks, causes or recommendations that are not in the data.\n"
                "- If not all rows are shown, say that the answer is based on a partial result.\n"
                "- If there are no rows, say that no matching data was found."
            ),
            config=types.GenerateContentConfig(max_output_tokens=512)
        )
    except genai_errors.APIError as e:
        # The SQL ran; only the explanation failed. Report it as a Gemini
        # problem and still show the results, so the user gets the data.
        return {
            "answer": f"The query ran, but Gemini was unavailable to explain the results ({e.code}). "
                      f"Columns: {cols}. First rows: {rows[:5]}",
            "sql": sql,
            "columns": cols,
            "rows": rows,
            "validation": "API_ERROR",
            "input_tokens": sql_result["input_tokens"],
            "output_tokens": sql_result["output_tokens"]
        }
    interp_input, interp_output = token_counts(interpretation)
    log(query, "quantitative-interpret", interp_input, interp_output)

    return {
        "answer": interpretation.text or "",
        "sql": sql,
        "columns": cols,
        "rows": rows,
        "validation": "PASSED",
        "input_tokens": sql_result["input_tokens"] + interp_input,
        "output_tokens": sql_result["output_tokens"] + interp_output
    }
