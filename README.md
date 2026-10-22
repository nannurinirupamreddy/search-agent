# Search Agent

An AI agent that answers questions using **live web search**, cites every claim, and checks that each citation points to a real source. Search providers (Tavily, Brave) and LLMs (Gemini, OpenAI) can be swapped, and a benchmark compares them on **latency** and **citation quality**.

```
$ python -m search_agent "When was the James Webb Space Telescope launched?" --verbose

<answer with inline citations like [1][2]>

Sources:
[1] <page title> - <url>
[2] <page title> - <url>

Searches: ['<query the agent chose>']
Latency: <total>s total (<search>s search, <llm>s LLM), grounded=True
```

## How it works

```
question ──► LLM (Gemini or OpenAI, via OpenAI-style tool calling)
                │  decides what to search
                ▼
          web_search tool ──► provider (Tavily | Brave) ──► cache (TTL)
                │  results numbered [1], [2], ... across all searches
                ▼
          LLM answers ONLY from results, citing [n]
                │
                ▼
          citation check ──► grounded? every [n] maps to a real source
```

- **Tool calling.** The model gets a single `web_search` tool and decides when and what to search. It can search again with a better query, up to `max_steps`. On the last step it's forced to answer.
- **Global source numbering.** Results are numbered across all searches, so `[3]` always means the same page.
- **Citation check.** After the answer comes back, every `[n]` is checked. An answer counts as **grounded** only if it has at least one citation and none of them point to a source that doesn't exist.
- **Two LLM backends, one code path.** Gemini's OpenAI-compatible endpoint means the same SDK and tool-calling loop run on either model. The backend is picked from whichever API key is set.
- **Provider interface.** Each search API is a small adapter that returns the same `SearchResult(title, url, snippet)`. Adding a new provider takes about 20 lines.
- **Reliability.** Retries with exponential backoff on HTTP 429 and 5xx, no retry on 4xx auth errors, and an in-memory TTL cache so repeated queries skip the network.

## Quick start

```bash
git clone <this repo> && cd search-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your keys
python -m search_agent "What is the Model Context Protocol?" --verbose
# force a backend: --llm gemini  |  --llm openai
```

| Variable | Required | Notes |
|---|---|---|
| `GEMINI_API_KEY` | one LLM key | Free tier at aistudio.google.com/apikey |
| `OPENAI_API_KEY` | one LLM key | Used instead of Gemini if set |
| `LLM_MODEL` | no | Defaults to `gemini-2.5-flash` or `gpt-4o-mini` |
| `TAVILY_API_KEY` | for `--provider tavily` | Free tier at tavily.com |
| `BRAVE_API_KEY` | for `--provider brave` | Free tier at brave.com/search/api |

## Use it in your own code

```python
from search_agent import SearchAgent, get_provider

agent = SearchAgent(get_provider("tavily"))
answer = agent.ask("What's the latest stable Python release?")

print(answer.format())            # text + numbered sources
print(answer.is_grounded)         # True if every citation is real
print(answer.total_latency_s, answer.search_latency_s, answer.llm_latency_s)
```

### Add a new search provider

Subclass `SearchProvider`, implement `_search`, and register it:

```python
class MyProvider(SearchProvider):
    name = "mine"

    def _search(self, query, max_results):
        r = self.client.get("https://api.example.com/search", params={"q": query})
        r.raise_for_status()
        return [SearchResult(x["title"], x["url"], x["snippet"]) for x in r.json()["items"]][:max_results]

PROVIDERS["mine"] = MyProvider
```

Caching, retries, and timing come from the base class.

## Benchmark

`benchmark/questions.json` holds 20 questions covering current facts, technical concepts, and general knowledge. The benchmark runs each one through the agent with each provider:

```bash
python benchmark/run_benchmark.py --providers tavily brave
```

It writes per-question rows to `benchmark/results.csv` and a summary table to `benchmark/results.md`:

| Metric | Meaning |
|---|---|
| Grounded answers | % of answers with at least one citation and no invalid ones |
| Median / p90 latency | End-to-end time per question (search + LLM) |
| Median per search | Time for one search API call |
| Avg searches | How many searches the agent needed per question |
| Invalid citations | Citations pointing to sources that don't exist |

### Results

*Run the benchmark with your keys and paste `benchmark/results.md` here.*

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

18 tests run fully offline, with the LLM and search APIs faked. They cover:

- The agent loop: search, answer, and multiple searches with shared numbering
- Catching made-up citations
- Forcing an answer on the last step
- Malformed tool arguments
- Cache hits
- Response parsing for each provider
- Retry on 429, no retry on 401, and a clear error when an API key is missing
- Picking the LLM backend from whichever key is set

CI runs them on every push with GitHub Actions.

## Project layout

```
search_agent/
  agent.py        # tool-calling loop, citation checking, timing
  providers.py    # Tavily + Brave adapters, retries, TTL cache
  __main__.py     # CLI
benchmark/
  questions.json  # 20 test questions
  run_benchmark.py
tests/
  test_agent.py
```

## Author

**Nirupam Reddy Nannuri** · [GitHub](https://github.com/nannurinirupamreddy) · [LinkedIn](https://linkedin.com/in/nirupamreddy)
