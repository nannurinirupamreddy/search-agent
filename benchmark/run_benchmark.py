"""Benchmark search providers on latency and citation quality.

Runs every question in questions.json through the agent once per provider,
then writes per-question results to CSV and a summary table to Markdown.

    python benchmark/run_benchmark.py --providers tavily brave
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from search_agent import SearchAgent, SearchError, get_provider  # noqa: E402

HERE = Path(__file__).resolve().parent


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run(provider_name: str, questions: list[str], model: str | None) -> list[dict]:
    agent = SearchAgent(get_provider(provider_name), model=model)
    rows = []
    for i, q in enumerate(questions, 1):
        try:
            a = agent.ask(q)
            rows.append(
                {
                    "provider": provider_name,
                    "question": q,
                    "total_s": round(a.total_latency_s, 3),
                    "search_s": round(a.search_latency_s, 3),
                    "llm_s": round(a.llm_latency_s, 3),
                    "searches": len(a.queries),
                    "sources": len(a.sources),
                    "citations": len(a.cited),
                    "invalid_citations": len(a.invalid_citations),
                    "grounded": a.is_grounded,
                    "error": "",
                }
            )
            print(f"[{provider_name} {i}/{len(questions)}] {a.total_latency_s:5.2f}s grounded={a.is_grounded}  {q[:60]}")
        except SearchError as e:
            rows.append({"provider": provider_name, "question": q, "error": str(e), "grounded": False})
            print(f"[{provider_name} {i}/{len(questions)}] ERROR {e}")
    return rows


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    totals = [r["total_s"] for r in ok]
    searches = [r["search_s"] / r["searches"] for r in ok if r["searches"]]
    return {
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "grounded_pct": round(100 * sum(r["grounded"] for r in rows) / len(rows), 1) if rows else 0,
        "median_total_s": round(statistics.median(totals), 2) if totals else 0,
        "p90_total_s": round(pct(totals, 90), 2),
        "median_per_search_s": round(statistics.median(searches), 3) if searches else 0,
        "avg_searches": round(statistics.mean(r["searches"] for r in ok), 2) if ok else 0,
        "invalid_citations": sum(r["invalid_citations"] for r in ok),
    }


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--providers", nargs="+", default=["tavily"])
    p.add_argument("--model", default=None)
    p.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    args = p.parse_args()

    questions = json.loads((HERE / "questions.json").read_text())[: args.limit]
    all_rows, summary = [], {}
    for name in args.providers:
        rows = run(name, questions, args.model)
        all_rows += rows
        summary[name] = summarize(rows)

    fields = ["provider", "question", "total_s", "search_s", "llm_s", "searches", "sources",
              "citations", "invalid_citations", "grounded", "error"]
    with open(HERE / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fields})

    lines = [
        "| Provider | Questions | Grounded answers | Median latency | p90 latency | Median per search | Avg searches | Invalid citations | Errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s['questions']} | {s['grounded_pct']}% | {s['median_total_s']}s | {s['p90_total_s']}s | "
            f"{s['median_per_search_s']}s | {s['avg_searches']} | {s['invalid_citations']} | {s['errors']} |"
        )
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
