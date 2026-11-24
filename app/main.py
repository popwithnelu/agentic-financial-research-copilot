from fastapi import FastAPI
from pydantic import BaseModel

from .agent import ResearchAgent

app = FastAPI(title="Agentic Financial Research Copilot")
agent = ResearchAgent()


class ResearchRequest(BaseModel):
    question: str


class CompareRequest(BaseModel):
    tickers: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "llm_mode": agent.llm.mode}


@app.get("/tickers")
def tickers():
    from .tools import MOCK_MARKET_DATA

    return {"tickers": sorted(MOCK_MARKET_DATA)}


@app.post("/research")
def research(req: ResearchRequest):
    return agent.research(req.question)


@app.post("/compare")
def compare(req: CompareRequest):
    return agent.compare(req.tickers)
