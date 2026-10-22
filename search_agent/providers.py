"""Web search providers behind one small interface.

Each provider turns a query into a list of SearchResult objects, so the agent
never needs to know which search API it is talking to. Swapping providers is a
one-line change, which is what makes side-by-side benchmarks possible.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult]
    latency_s: float
    cached: bool = False


class SearchError(RuntimeError):
    """Raised when a provider returns an error the agent can't recover from."""


@dataclass
class _Cache:
    """Tiny in-memory TTL cache so repeated queries skip the network."""

    ttl_s: float = 600.0
    _store: dict[tuple[str, int], tuple[float, list[SearchResult]]] = field(default_factory=dict)

    def get(self, key: tuple[str, int]) -> list[SearchResult] | None:
        hit = self._store.get(key)
        if hit and time.monotonic() - hit[0] < self.ttl_s:
            return hit[1]
        return None

    def put(self, key: tuple[str, int], value: list[SearchResult]) -> None:
        self._store[key] = (time.monotonic(), value)


class SearchProvider:
    name = "base"

    def __init__(self, client: httpx.Client | None = None, max_retries: int = 2, cache_ttl_s: float = 600.0):
        self.client = client or httpx.Client(timeout=15.0)
        self.max_retries = max_retries
        self.cache = _Cache(ttl_s=cache_ttl_s)

    def search(self, query: str, max_results: int = 5) -> SearchResponse:
        key = (query.strip().lower(), max_results)
        start = time.perf_counter()
        cached = self.cache.get(key)
        if cached is not None:
            return SearchResponse(query, cached, time.perf_counter() - start, cached=True)

        results = self._with_retries(lambda: self._search(query, max_results))
        self.cache.put(key, results)
        return SearchResponse(query, results, time.perf_counter() - start)

    def _with_retries(self, fn):
        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                return fn()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                retryable = status == 429 or status >= 500
                if not retryable or attempt == self.max_retries:
                    raise SearchError(f"{self.name} returned HTTP {status}") from e
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise SearchError(f"{self.name} unreachable: {e}") from e
            time.sleep(delay)
            delay *= 2
        raise SearchError(f"{self.name} failed")  # pragma: no cover

    def _search(self, query: str, max_results: int) -> list[SearchResult]:  # pragma: no cover
        raise NotImplementedError


class TavilyProvider(SearchProvider):
    """https://docs.tavily.com - search API built for LLM agents."""

    name = "tavily"
    url = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, **kw):
        super().__init__(**kw)
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise SearchError("Set TAVILY_API_KEY to use the Tavily provider")

    def _search(self, query: str, max_results: int) -> list[SearchResult]:
        r = self.client.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"query": query, "max_results": max_results, "search_depth": "basic"},
        )
        r.raise_for_status()
        return [
            SearchResult(title=x.get("title", ""), url=x["url"], snippet=x.get("content", ""))
            for x in r.json().get("results", [])
            if x.get("url")
        ][:max_results]


class BraveProvider(SearchProvider):
    """https://api-dashboard.search.brave.com - independent web index."""

    name = "brave"
    url = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None, **kw):
        super().__init__(**kw)
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY")
        if not self.api_key:
            raise SearchError("Set BRAVE_API_KEY to use the Brave provider")

    def _search(self, query: str, max_results: int) -> list[SearchResult]:
        r = self.client.get(
            self.url,
            headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
            params={"q": query, "count": max_results},
        )
        r.raise_for_status()
        web = r.json().get("web", {}).get("results", [])
        return [
            SearchResult(title=x.get("title", ""), url=x["url"], snippet=x.get("description", ""))
            for x in web
            if x.get("url")
        ][:max_results]


PROVIDERS = {"tavily": TavilyProvider, "brave": BraveProvider}


def get_provider(name: str, **kw) -> SearchProvider:
    try:
        return PROVIDERS[name](**kw)
    except KeyError:
        raise SearchError(f"Unknown provider {name!r}; choose from {sorted(PROVIDERS)}") from None
