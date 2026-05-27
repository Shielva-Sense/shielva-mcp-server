"""Adapter wrapping the existing HybridRetriever (RRF over dense +
keyword) into :class:`domain.knowledge.Retriever`.

The existing retriever already implements RRF. We translate its
``RetrievalResult`` shape (chunks + metadata dicts) into our domain
:class:`RetrievedChunk` value objects so the application layer sees
a clean type.
"""
from __future__ import annotations

from typing import List

from src.domain.knowledge.repositories import Retriever
from src.domain.knowledge.value_objects import (
    Chunk, ChunkId, DocumentId, KBId, RetrievalQuery, RetrievedChunk,
)
from src.domain.shared.tenant import TenantContext


class LegacyHybridRetrieverAdapter(Retriever):
    def __init__(self, legacy_retriever) -> None:
        # ``src.rag_engine.retriever.hybrid_retriever.HybridRetriever``
        self._retriever = legacy_retriever

    async def retrieve(
        self,
        query: RetrievalQuery,
        *,
        tenant: TenantContext,
    ) -> List[RetrievedChunk]:
        # The legacy retriever takes a single kb_id per call. For
        # multi-KB queries we fan out and merge.
        merged: List[RetrievedChunk] = []
        for kb_id in query.kb_ids:
            results = await self._retriever.retrieve(
                query=query.query,
                tenant_id=tenant.tenant_id,
                kb_id=str(kb_id),
                top_k=query.top_k,
            )
            for r in (results or []):
                merged.append(_to_retrieved(r, kb_id))
        # Final cross-KB sort + cap.
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged[: query.top_k]


def _to_retrieved(r, kb_id: KBId) -> RetrievedChunk:
    """Translate ``RetrievalResult`` (legacy dataclass) → domain VO."""
    metadata = dict(getattr(r, "metadata", {}) or {})
    chunk = Chunk(
        id          = ChunkId(str(getattr(r, "chunk_id", "")
                                  or metadata.get("chunk_id", ""))),
        document_id = DocumentId(str(getattr(r, "document_id", "")
                                     or metadata.get("document_id", ""))),
        kb_id       = kb_id,
        content     = str(getattr(r, "content", "") or ""),
        metadata    = metadata,
    )
    return RetrievedChunk(chunk=chunk, score=float(getattr(r, "score", 0.0) or 0.0))
