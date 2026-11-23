"""Tools the research agent can call (LangChain-style tool interface)."""

MOCK_MARKET_DATA = {
    "ACME": {"price": 182.40, "pe_ratio": 24.1, "revenue_growth_pct": 12.5, "eps": 7.57},
    "GLOBEX": {"price": 64.20, "pe_ratio": 15.8, "revenue_growth_pct": 6.1, "eps": 4.06},
    "INITECH": {"price": 41.75, "pe_ratio": 31.4, "revenue_growth_pct": 18.9, "eps": 1.33},
}


class MarketDataTool:
    name = "market_data"
    description = "Look up mock market data (price, P/E, revenue growth, EPS) for a ticker."

    def run(self, ticker: str) -> dict:
        data = MOCK_MARKET_DATA.get(ticker.upper())
        if data is None:
            return {"ticker": ticker.upper(), "error": "unknown ticker"}
        return {"ticker": ticker.upper(), **data}


class DocumentSearchTool:
    name = "document_search"
    description = "Search the research document corpus for a query."

    def __init__(self, store):
        self.store = store

    def run(self, query: str) -> list[dict]:
        return [{"title": d["title"], "text": d["text"]} for d in self.store.search(query)]
