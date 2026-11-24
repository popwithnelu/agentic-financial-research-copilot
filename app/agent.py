"""Multi-step research agent: plan -> gather (tools) -> synthesize (LLM)."""
from .documents import DocumentStore
from .llm import LLMClient
from .tools import DocumentSearchTool, MarketDataTool


class ResearchAgent:
    def __init__(self, llm: LLMClient | None = None, store: DocumentStore | None = None):
        self.llm = llm or LLMClient()
        self.store = store or DocumentStore.from_file()
        self.market = MarketDataTool()
        self.docs = DocumentSearchTool(self.store)

    def plan(self, question: str) -> list[str]:
        steps = ["search documents for background"]
        for ticker in self.market_tickers(question):
            steps.append(f"fetch market data for {ticker}")
        steps.append("synthesize answer with the LLM")
        return steps

    @staticmethod
    def market_tickers(question: str) -> list[str]:
        from .tools import MOCK_MARKET_DATA

        words = {w.strip(".,?").upper() for w in question.split()}
        return sorted(words & set(MOCK_MARKET_DATA))

    def research(self, question: str) -> dict:
        steps = self.plan(question)
        evidence = {"documents": self.docs.run(question), "market_data": []}
        for ticker in self.market_tickers(question):
            evidence["market_data"].append(self.market.run(ticker))
        prompt = (
            f"Question: {question}\n"
            f"Document evidence: {evidence['documents']}\n"
            f"Market data: {evidence['market_data']}\n"
            "Write a concise research answer grounded only in the evidence above."
        )
        answer = self.llm.complete(prompt)
        return {"question": question, "steps": steps, "evidence": evidence,
                "answer": answer, "llm_mode": self.llm.mode}

    def compare(self, tickers: list[str]) -> dict:
        """Compare tickers side by side and highlight simple leaders.

        Returns the raw market data for each known ticker plus a small
        summary naming the cheapest (lowest P/E), fastest-growing, and
        highest-priced ticker among those compared — useful context a
        researcher would otherwise have to compute by hand.
        """
        rows = [self.market.run(t) for t in tickers]
        known = [r for r in rows if "error" not in r]
        summary: dict = {}
        if known:
            summary = {
                "lowest_pe": min(known, key=lambda r: r["pe_ratio"])["ticker"],
                "highest_revenue_growth": max(
                    known, key=lambda r: r["revenue_growth_pct"]
                )["ticker"],
                "highest_price": max(known, key=lambda r: r["price"])["ticker"],
            }
        return {"tickers": rows, "summary": summary}
