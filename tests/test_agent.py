"""Tests run fully offline: the OpenAI client and search APIs are faked."""

import json
from types import SimpleNamespace as NS

import httpx
import pytest
import respx

from search_agent import SearchAgent, SearchError, SearchResult, check_citations
from search_agent.providers import BraveProvider, SearchProvider, SearchResponse, TavilyProvider

# ---------- fakes ----------


class FakeProvider(SearchProvider):
    name = "fake"

    def __init__(self, results=None):
        super().__init__(client=httpx.Client())
        self.calls = []
        self.results = results or [
            SearchResult("Python 3.14 released", "https://python.org/a", "Python 3.14 is out."),
            SearchResult("Release notes", "https://docs.python.org/b", "What's new in 3.14."),
        ]

    def _search(self, query, max_results):
        self.calls.append(query)
        return self.results[:max_results]


def tool_call(query, id="call_1"):
    return NS(id=id, type="function", function=NS(name="web_search", arguments=json.dumps({"query": query})))


def completion(content=None, tool_calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))])


class FakeOpenAI:
    """Replays a scripted list of completions and records each request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        return self.script.pop(0)


# ---------- citations ----------


def test_check_citations_valid_and_invalid():
    cited, invalid = check_citations("A [1]. B [2][3]. C [7].", n_sources=3)
    assert cited == [1, 2, 3, 7]
    assert invalid == [7]


def test_check_citations_none():
    assert check_citations("No sources here.", 3) == ([], [])


# ---------- agent loop ----------


def test_agent_searches_then_answers_with_valid_citations():
    provider = FakeProvider()
    llm = FakeOpenAI([
        completion(tool_calls=[tool_call("latest python version")]),
        completion(content="Python 3.14 is the latest release [1][2]."),
    ])
    ans = SearchAgent(provider, client=llm, model="test").ask("What is the latest Python?")

    assert provider.calls == ["latest python version"]
    assert ans.queries == ["latest python version"]
    assert len(ans.sources) == 2
    assert ans.cited == [1, 2]
    assert ans.is_grounded
    assert "[1] Python 3.14 released - https://python.org/a" in ans.format()

    # The tool result sent back to the model carries numbered sources.
    tool_msg = llm.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool"
    assert json.loads(tool_msg["content"])["results"][0]["source"] == 1


def test_agent_flags_hallucinated_citation():
    llm = FakeOpenAI([
        completion(tool_calls=[tool_call("q")]),
        completion(content="Claim [1]. Made-up claim [9]."),
    ])
    ans = SearchAgent(FakeProvider(), client=llm, model="test").ask("q")
    assert ans.invalid_citations == [9]
    assert not ans.is_grounded


def test_agent_answer_without_citations_is_not_grounded():
    llm = FakeOpenAI([completion(content="I think it's 3.14.")])
    ans = SearchAgent(FakeProvider(), client=llm, model="test").ask("q")
    assert ans.sources == [] and not ans.is_grounded


def test_agent_numbers_sources_across_multiple_searches():
    llm = FakeOpenAI([
        completion(tool_calls=[tool_call("first", "c1")]),
        completion(tool_calls=[tool_call("second", "c2")]),
        completion(content="Answer [3]."),
    ])
    ans = SearchAgent(FakeProvider(), client=llm, model="test").ask("q")
    assert len(ans.sources) == 4  # 2 results x 2 searches, numbered 1..4
    assert ans.is_grounded


def test_agent_forces_answer_on_last_step():
    llm = FakeOpenAI([
        completion(tool_calls=[tool_call("a", "c1")]),
        completion(content="Done [1]."),
    ])
    SearchAgent(FakeProvider(), client=llm, model="test", max_steps=2).ask("q")
    assert llm.requests[0]["tool_choice"] == "auto"
    assert llm.requests[-1]["tool_choice"] == "none"


def test_agent_handles_bad_tool_arguments():
    bad = NS(id="c1", type="function", function=NS(name="web_search", arguments="not json"))
    llm = FakeOpenAI([completion(tool_calls=[bad]), completion(content="Could not search.")])
    provider = FakeProvider()
    ans = SearchAgent(provider, client=llm, model="test").ask("q")
    assert provider.calls == []
    assert "error" in json.loads(llm.requests[1]["messages"][-1]["content"])
    assert not ans.is_grounded


# ---------- caching ----------


def test_repeated_query_is_served_from_cache():
    provider = FakeProvider()
    first = provider.search("Same Query")
    second = provider.search("  same query ")
    assert provider.calls == ["Same Query"]
    assert not first.cached and second.cached


def test_cache_hits_reported_by_agent():
    provider = FakeProvider()
    llm = FakeOpenAI([
        completion(tool_calls=[tool_call("dup", "c1"), tool_call("dup", "c2")]),
        completion(content="Answer [1]."),
    ])
    ans = SearchAgent(provider, client=llm, model="test").ask("q")
    assert provider.calls == ["dup"]
    assert ans.cache_hits == 1


# ---------- real provider adapters (HTTP mocked) ----------


@respx.mock
def test_tavily_parses_results():
    respx.post(TavilyProvider.url).mock(return_value=httpx.Response(200, json={
        "results": [{"title": "T", "url": "https://t.com", "content": "snippet"}, {"title": "no url"}]
    }))
    res = TavilyProvider(api_key="k").search("q")
    assert [r.url for r in res.results] == ["https://t.com"]
    assert res.results[0].snippet == "snippet"


@respx.mock
def test_brave_parses_results():
    respx.get(BraveProvider.url).mock(return_value=httpx.Response(200, json={
        "web": {"results": [{"title": "B", "url": "https://b.com", "description": "d"}]}
    }))
    res = BraveProvider(api_key="k").search("q")
    assert res.results[0].title == "B" and res.results[0].snippet == "d"


@respx.mock
def test_provider_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr("search_agent.providers.time.sleep", lambda s: None)
    route = respx.post(TavilyProvider.url).mock(side_effect=[
        httpx.Response(429),
        httpx.Response(200, json={"results": [{"title": "T", "url": "https://t.com", "content": ""}]}),
    ])
    res = TavilyProvider(api_key="k").search("q")
    assert route.call_count == 2 and len(res.results) == 1


@respx.mock
def test_provider_does_not_retry_on_401(monkeypatch):
    monkeypatch.setattr("search_agent.providers.time.sleep", lambda s: None)
    route = respx.post(TavilyProvider.url).mock(return_value=httpx.Response(401))
    with pytest.raises(SearchError, match="401"):
        TavilyProvider(api_key="k").search("q")
    assert route.call_count == 1


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(SearchError, match="TAVILY_API_KEY"):
        TavilyProvider()


# ---------- LLM backend selection ----------


def test_backend_autodetects_gemini(monkeypatch):
    from search_agent.agent import GEMINI_BASE_URL, make_llm_client

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    client, backend = make_llm_client()
    assert backend == "gemini"
    assert str(client.base_url) == GEMINI_BASE_URL


def test_backend_requires_a_key(monkeypatch):
    from search_agent.agent import make_llm_client

    for v in ("OPENAI_API_KEY", "GEMINI_API_KEY", "LLM_BACKEND"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        make_llm_client()


def test_default_model_follows_backend(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    agent = SearchAgent(FakeProvider(), client=FakeOpenAI([]), backend="gemini")
    assert agent.model == "gemini-2.5-flash"
