"""Command line: python -m search_agent "your question" [--provider tavily|brave]"""

import argparse
import sys

from dotenv import load_dotenv

from .agent import SearchAgent
from .providers import PROVIDERS, SearchError, get_provider


def main() -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Answer a question with live web search and citations.")
    p.add_argument("question")
    p.add_argument("--provider", choices=sorted(PROVIDERS), default="tavily")
    p.add_argument("--model", default=None, help="OpenAI model (default: $OPENAI_MODEL or gpt-4o-mini)")
    p.add_argument("--verbose", action="store_true", help="Show searches and timing")
    args = p.parse_args()

    try:
        agent = SearchAgent(get_provider(args.provider), model=args.model)
        ans = agent.ask(args.question)
    except SearchError as e:
        print(f"Search error: {e}", file=sys.stderr)
        return 1

    print(ans.format())
    if args.verbose:
        print(f"\nSearches: {ans.queries}")
        print(
            f"Latency: {ans.total_latency_s:.2f}s total "
            f"({ans.search_latency_s:.2f}s search, {ans.llm_latency_s:.2f}s LLM), "
            f"grounded={ans.is_grounded}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
