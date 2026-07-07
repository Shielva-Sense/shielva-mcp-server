"""Adapter wrapping the existing ``src.embedder.EmbeddingClient``
into the new :class:`domain.knowledge.EmbeddingClient` port.

Same Gemini / OpenAI / Cohere fan-out under the hood — the adapter
just translates dataclass shapes and exposes ``dimension`` / ``model``
as port-level properties for tools that need to size collections.
"""

from __future__ import annotations

from src.domain.knowledge.repositories import EmbeddingClient


class LegacyEmbeddingClientAdapter(EmbeddingClient):
    def __init__(self, legacy_client) -> None:
        # ``legacy_client`` is the existing ``src.embedder.EmbeddingClient``
        # instance constructed in main.py lifespan.
        self._legacy = legacy_client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._legacy.embed(texts)

    @property
    def dimension(self) -> int:
        return int(getattr(self._legacy.config, "dimension", 0) or 0)

    @property
    def model(self) -> str:
        return str(getattr(self._legacy.config, "model", ""))
