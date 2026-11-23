from fastapi import FastAPI
from pydantic import BaseModel

from .agent import ResearchAgent

app = FastAPI(title="Agentic Financial Research Copilot")
agent = ResearchAgent()


class ResearchRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "llm_mode": agent.llm.mode}


@app.post("/research")
def research(req: ResearchRequest):
    return agent.research(req.question)
