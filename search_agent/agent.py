"""An LLM agent that answers questions using live web search, with citations.

Flow:
  1. The model gets the question and a `web_search` tool (OpenAI tool calling).
  2. It decides what to search; each result gets a global source number [n].
  3. It answers using only those sources, citing them inline as [n].
  4. We check every citation points at a real source before returning.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from openai import OpenAI

from .providers import SearchProvider, SearchResult

SYSTEM_PROMPT = """You are a research assistant that answers using live web search.

Rules:
- Call web_search for anything that depends on facts, dates, or recent events. Search again with a better query if results are weak.
- Answer ONLY from the search results. Cite every factual sentence with its source number in square brackets, like [1] or [2][3].
- Only cite numbers that appeared in search results. Never invent sources.
- If the results don't answer the question, say so plainly instead of guessing.
- Keep answers concise: 2-5 sentences unless the user asks for more."""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the live web. Returns numbered results with title, URL, and snippet.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused search query."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

CITATION_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Answer:
    question: str
    text: str
    sources: list[SearchResult]
    queries: list[str]
    total_latency_s: float
    search_latency_s: float
    llm_latency_s: float
    cache_hits: int = 0
    cited: list[int] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        """At least one citation, and every citation points at a real source."""
        return bool(self.cited) and not self.invalid_citations

    def format(self) -> str:
        used = [i for i in sorted(set(self.cited)) if 1 <= i <= len(self.sources)]
        lines = [self.text, ""]
        if used:
            lines.append("Sources:")
            lines += [f"[{i}] {self.sources[i - 1].title} - {self.sources[i - 1].url}" for i in used]
        return "\n".join(lines)


def check_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    cited = [int(m) for m in CITATION_RE.findall(text)]
    invalid = sorted({c for c in cited if c < 1 or c > n_sources})
    return cited, invalid


class SearchAgent:
    def __init__(
        self,
        provider: SearchProvider,
        client: OpenAI | None = None,
        model: str | None = None,
        max_steps: int = 4,
        results_per_search: int = 5,
    ):
        self.provider = provider
        self.client = client or OpenAI()
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.max_steps = max_steps
        self.results_per_search = results_per_search

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        search_s = llm_s = 0.0
        cache_hits = 0
        sources: list[SearchResult] = []
        queries: list[str] = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        text = ""
        for step in range(self.max_steps):
            last = step == self.max_steps - 1
            t = time.perf_counter()
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[SEARCH_TOOL],
                # On the final step, force an answer instead of another search.
                tool_choice="none" if last else "auto",
                temperature=0,
            )
            llm_s += time.perf_counter() - t
            msg = resp.choices[0].message

            if not msg.tool_calls:
                text = (msg.content or "").strip()
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                        for c in msg.tool_calls
                    ],
                }
            )
            for call in msg.tool_calls:
                content, latency, cached = self._run_tool(call, sources, queries)
                search_s += latency
                cache_hits += int(cached)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

        cited, invalid = check_citations(text, len(sources))
        return Answer(
            question=question,
            text=text,
            sources=sources,
            queries=queries,
            total_latency_s=time.perf_counter() - start,
            search_latency_s=search_s,
            llm_latency_s=llm_s,
            cache_hits=cache_hits,
            cited=cited,
            invalid_citations=invalid,
        )

    def _run_tool(self, call, sources: list[SearchResult], queries: list[str]) -> tuple[str, float, bool]:
        """Run one tool call. Returns (tool message content, search latency, cache hit)."""
        if call.function.name != "web_search":
            return json.dumps({"error": f"unknown tool {call.function.name}"}), 0.0, False
        try:
            query = json.loads(call.function.arguments or "{}")["query"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return json.dumps({"error": "web_search needs a 'query' string"}), 0.0, False

        queries.append(query)
        resp = self.provider.search(query, self.results_per_search)

        numbered = []
        for r in resp.results:
            sources.append(r)
            numbered.append({"source": len(sources), "title": r.title, "url": r.url, "snippet": r.snippet[:600]})
        if not numbered:
            content = json.dumps({"results": [], "note": "No results. Try a different query."})
        else:
            content = json.dumps({"results": numbered})
        return content, resp.latency_s, resp.cached
