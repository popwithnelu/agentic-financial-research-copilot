# Agentic Financial Research Copilot

A multi-step AI research agent that answers financial questions by planning,
calling tools (mock market data + document search over a small corpus), and
synthesizing a grounded answer with an LLM.

## Architecture

```
question -> ResearchAgent.plan() -> DocumentSearchTool (RAG over sample docs)
                                 -> MarketDataTool (mock quotes/fundamentals)
                                 -> LLMClient (OpenAI / Claude / mock) -> answer + evidence
```

- `app/agent.py` — plan → gather → synthesize loop (LangChain-style tool use, no framework lock-in)
- `app/tools.py` — tools with a uniform `run()` interface
- `app/documents.py` — keyword-overlap retrieval over `data/sample_docs.json`
- `app/llm.py` — OpenAI/Claude wrappers; falls back to a deterministic mock model offline
- `app/main.py` — FastAPI service (`/health`, `/research`, `/tickers`, `/compare`)

## Tech stack
Python, FastAPI, Pydantic; OpenAI API / Claude API (optional, live mode only).

## Run

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
# offline mock mode works with no keys; for live mode set OPENAI_API_KEY or ANTHROPIC_API_KEY
curl -X POST localhost:8000/research -H 'Content-Type: application/json' \
  -d '{"question": "How is ACME growing?"}'

# Compare tickers side by side (lowest P/E, fastest growth, highest price):
curl localhost:8000/tickers
curl -X POST localhost:8000/compare -H 'Content-Type: application/json' \
  -d '{"tickers": ["ACME", "GLOBEX", "INITECH"]}'
```

## Tests

```bash
python -m pytest   # runs fully offline in mock mode
```

## Project structure

```
app/ (agent, tools, documents, llm, main) · data/sample_docs.json · tests/test_agent.py
```

MIT License — see LICENSE.
