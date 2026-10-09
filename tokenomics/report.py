# tokenomics/report.py
"""Summarise tokenomics_log.jsonl: run after each test session to spot expensive calls.

Usage:  python -m tokenomics.report          # whole log
        python -m tokenomics.report --top 10  # show the 10 most expensive calls
"""
import json
import sys
from collections import defaultdict
from tokenomics.logger import LOG_PATH

def load_entries() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    top_n = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 5
    entries = load_entries()
    if not entries:
        print(f"No entries in {LOG_PATH}")
        return

    by_agent = defaultdict(lambda: {"calls": 0, "input": 0, "output": 0, "cost": 0.0})
    for e in entries:
        a = by_agent[e["agent"]]
        a["calls"] += 1
        a["input"] += e["input_tokens"]
        a["output"] += e["output_tokens"]
        a["cost"] += e["cost_usd"]

    print(f"Log: {LOG_PATH}")
    print(f"Period: {entries[0]['timestamp'][:19]} to {entries[-1]['timestamp'][:19]}\n")

    header = f"{'Agent':<24}{'Calls':>6}{'Input':>10}{'Output':>10}{'Avg in':>9}{'Avg out':>9}{'Cost $':>11}"
    print(header)
    print("-" * len(header))
    for agent, a in sorted(by_agent.items(), key=lambda kv: -kv[1]["cost"]):
        print(f"{agent:<24}{a['calls']:>6}{a['input']:>10,}{a['output']:>10,}"
              f"{a['input'] // a['calls']:>9,}{a['output'] // a['calls']:>9,}{a['cost']:>11.6f}")

    total_cost = sum(a["cost"] for a in by_agent.values())
    total_in = sum(a["input"] for a in by_agent.values())
    total_out = sum(a["output"] for a in by_agent.values())
    print("-" * len(header))
    print(f"{'Total':<24}{len(entries):>6}{total_in:>10,}{total_out:>10,}{'':>18}{total_cost:>11.6f}")

    # One user question can trigger several calls (classifier, SQL, interpretation...).
    queries = {e["query"] for e in entries}
    per_query = total_cost / len(queries)
    print(f"\nDistinct queries: {len(queries)}")
    print(f"Average cost per query: ${per_query:.6f}  (≈ ${per_query * 1000:.2f} per 1,000 queries)")

    print(f"\nMost expensive calls:")
    for e in sorted(entries, key=lambda e: -e["cost_usd"])[:top_n]:
        print(f"  ${e['cost_usd']:.6f}  {e['agent']:<22} in={e['input_tokens']:<6} out={e['output_tokens']:<6} {e['query'][:60]}")

if __name__ == "__main__":
    main()
