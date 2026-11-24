from fastapi.testclient import TestClient

from app.agent import ResearchAgent
from app.main import app
from app.tools import MarketDataTool


def test_market_data_tool_known_and_unknown():
    tool = MarketDataTool()
    assert tool.run("ACME")["price"] == 182.40
    assert "error" in tool.run("NOPE")


def test_agent_research_offline():
    result = ResearchAgent().research("How is ACME growing and what are its margins?")
    assert result["llm_mode"] == "mock"
    assert result["evidence"]["documents"], "expected document evidence"
    assert result["evidence"]["market_data"][0]["ticker"] == "ACME"
    assert result["answer"]


def test_api_health_and_research():
    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    resp = client.post("/research", json={"question": "Tell me about GLOBEX valuation"})
    assert resp.status_code == 200
    assert resp.json()["evidence"]["market_data"][0]["ticker"] == "GLOBEX"


def test_agent_compare_tickers():
    result = ResearchAgent().compare(["ACME", "GLOBEX", "INITECH"])
    assert [r["ticker"] for r in result["tickers"]] == ["ACME", "GLOBEX", "INITECH"]
    # GLOBEX has the lowest P/E (15.8); INITECH grows fastest (18.9%); ACME is priciest.
    assert result["summary"] == {
        "lowest_pe": "GLOBEX",
        "highest_revenue_growth": "INITECH",
        "highest_price": "ACME",
    }


def test_agent_compare_with_unknown_ticker():
    result = ResearchAgent().compare(["ACME", "NOPE"])
    assert "error" in result["tickers"][1]
    assert result["summary"]["lowest_pe"] == "ACME"


def test_api_tickers_and_compare():
    client = TestClient(app)
    assert client.get("/tickers").json() == {"tickers": ["ACME", "GLOBEX", "INITECH"]}
    resp = client.post("/compare", json={"tickers": ["ACME", "GLOBEX"]})
    assert resp.status_code == 200
    assert resp.json()["summary"]["highest_revenue_growth"] == "ACME"
