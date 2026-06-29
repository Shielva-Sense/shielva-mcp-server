"""Adapter mapping the existing pgvector vector store onto the
:class:`domain.knowledge.VectorStore` port.

The legacy ``PgVectorStore`` carries a different vocabulary
(``collection_name = ns_{tenant}_{kb}``, dict-based ``upsert``).
This adapter does the value-object ↔ dict translation so domain
code stays clean.
"""
from __future__ import annotations

from typing import List, Tuple

import structlog

from src.domain.knowledge.repositories import VectorStore
from src.domain.knowledge.value_objects import (
    Chunk, ChunkId, DocumentId, KBId, RetrievedChunk,
)
from src.domain.shared.tenant import TenantContext

logger = structlog.get_logger(__name__)


class PgVectorStoreAdapter(VectorStore):
    def __init__(self, legacy_store) -> None:
        # Existing ``src.rag_engine.vectorstore.supabase_store.PgVectorStore``
        # instance from main.py lifespan.
        self._store = legacy_store

    async def upsert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        *,
        tenant: TenantContext,
    ) -> int:
        if not chunks:
            return 0
        # Legacy upsert is per-KB; group by kb_id.
        groups: dict[str, list[tuple[Chunk, list[float]]]] = {}
        for chunk, emb in zip(chunks, embeddings):
            groups.setdefault(str(chunk.kb_id), []).append((chunk, emb))
        total = 0
        for kb_id, items in groups.items():
            payload = [
                {
                    "id":          str(c.id),
                    "document_id": str(c.document_id),
                    "content":     c.content,
                    "metadata":    c.metadata,
                    "embedding":   emb,
                }
                for c, emb in items
            ]
            count = await self._store.upsert(
                tenant_id=tenant.tenant_id, kb_id=kb_id, chunks=payload,
            )
            total += int(count or 0)
        return total

    async def similarity_search(
        self,
        embedding: List[float],
        *,
        kb_ids: Tuple[KBId, ...],
        tenant: TenantContext,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        for kb_id in kb_ids:
            rows = await self._store.search(
                tenant_id=tenant.tenant_id,
                kb_id=str(kb_id),
                query_embedding=embedding,
                top_k=top_k,
            )
            out.extend(_to_retrieved(rows, kb_id))
        out.sort(key=lambda r: r.score, reverse=True)
        return out[:top_k]

    async def keyword_search(
        self,
        query: str,
        *,
        kb_ids: Tuple[KBId, ...],
        tenant: TenantContext,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        out: List[RetrievedChunk] = []
        for kb_id in kb_ids:
            rows = await self._store.keyword_search(
                tenant_id=tenant.tenant_id,
                kb_id=str(kb_id),
                query=query,
                top_k=top_k,
            )
            out.extend(_to_retrieved(rows, kb_id))
        out.sort(key=lambda r: r.score, reverse=True)
        return out[:top_k]

    async def delete_chunks(
        self,
        chunk_ids: List[ChunkId],
        *,
        tenant: TenantContext,
    ) -> int:
        # Legacy delete_by_ids is per-(tenant, kb) — we don't have
        # kb_id at the call site, so we delegate to the broader API
        # only if available. Adapters that don't support id-only
        # deletion may raise NotImplementedError instead.
        delete_by_ids = getattr(self._store, "delete_by_ids", None)
        if delete_by_ids is None:
            raise NotImplementedError(
                "Underlying store does not support id-only deletion"
            )
        return int(await delete_by_ids(
            tenant_id=tenant.tenant_id,
            chunk_ids=[str(c) for c in chunk_ids],
        ) or 0)


def _to_retrieved(rows, kb_id: KBId) -> List[RetrievedChunk]:
    """Translate provider rows (dict-shaped from legacy store) into
    domain ``RetrievedChunk`` value objects."""
    out: List[RetrievedChunk] = []
    for row in (rows or []):
        chunk = Chunk(
            id          = ChunkId(str(row.get("id") or row.get("chunk_id") or "")),
            document_id = DocumentId(str(row.get("document_id") or "")),
            kb_id       = kb_id,
            content     = str(row.get("content") or ""),
            metadata    = dict(row.get("metadata") or {}),
        )
        score = float(row.get("score") or row.get("similarity") or 0.0)
        out.append(RetrievedChunk(chunk=chunk, score=score))
    return out


# Backward-compat alias: renamed SupabaseVectorStoreAdapter -> PgVectorStoreAdapter
# (the underlying store is plain pgvector, not Supabase).
SupabaseVectorStoreAdapter = PgVectorStoreAdapter
