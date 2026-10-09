# tokenomics/logger.py
import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "tokenomics_log.jsonl"

# gemini-3.5-flash-lite, standard paid tier (checked 2026-10-08):
# $0.30 per 1M input tokens, $2.50 per 1M output tokens (thinking tokens included).
# Source: https://ai.google.dev/gemini-api/docs/pricing
COST_PER_1K_INPUT  = 0.0003
COST_PER_1K_OUTPUT = 0.0025

def token_counts(response) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from a Gemini response.

    Gemini reports "thinking" tokens separately from the visible answer, but both
    are billed as output, so they are added together here.
    """
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count or 0
    output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    return input_tokens, output_tokens

# The question the user typed for the current turn. The manager sets it at the
# start of each turn, so every Gemini call in that turn is logged under the same
# question, even when an agent is working on a split sub-question or a
# rewritten follow-up. Without this, one "both" question was logged under three
# different query strings and the report's cost-per-query was wrong.
_current_query = None

def set_current_query(query):
    global _current_query
    _current_query = query

def log(query: str, agent: str, input_tokens: int, output_tokens: int):
    input_cost  = (input_tokens  / 1000) * COST_PER_1K_INPUT
    output_cost = (output_tokens / 1000) * COST_PER_1K_OUTPUT
    total_cost  = input_cost + output_cost

    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": _current_query or query,
        "agent": agent,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(total_cost, 6),
        "cost_per_1000_queries": round(total_cost * 1000, 2)
    }
    # Keep the question this call actually worked on, when it differs.
    if _current_query and query != _current_query:
        entry["sub_query"] = query

    print(f"\n[TOKENOMICS] Agent: {agent} | Input: {input_tokens} | Output: {output_tokens} | Cost: ${total_cost:.6f}")

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry
