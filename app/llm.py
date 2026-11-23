"""LLM client wrappers with graceful offline/mock mode.

If OPENAI_API_KEY or ANTHROPIC_API_KEY is set (and the matching SDK is
installed), completions go to the real API. Otherwise a deterministic
mock model is used so the agent, API and tests run fully offline.
"""
import os


class MockLLM:
    """Deterministic stand-in that echoes a structured summary of the prompt."""

    name = "mock"

    def complete(self, prompt: str) -> str:
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        topic = lines[0][:120] if lines else "the request"
        return f"Mock analysis of {topic}: based on the gathered evidence, the key drivers are revenue growth, margins and guidance."


class LLMClient:
    def __init__(self):
        self.provider = None
        self._client = None
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic  # type: ignore

                self._client = anthropic.Anthropic()
                self.provider = "anthropic"
            except Exception:
                self._client = None
        if self._client is None and os.getenv("OPENAI_API_KEY"):
            try:
                import openai  # type: ignore

                self._client = openai.OpenAI()
                self.provider = "openai"
            except Exception:
                self._client = None
        self._mock = MockLLM()

    @property
    def mode(self) -> str:
        return self.provider or "mock"

    def complete(self, prompt: str, model: str | None = None) -> str:
        if self.provider == "anthropic":
            msg = self._client.messages.create(
                model=model or "claude-sonnet-4-5",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in msg.content if hasattr(b, "text"))
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""
        return self._mock.complete(prompt)
