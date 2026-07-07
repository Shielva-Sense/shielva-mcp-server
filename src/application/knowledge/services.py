"""Knowledge application service — listings + retrieval use cases."""

from __future__ import annotations

import time

import structlog

from src.domain.knowledge.entities import KnowledgeBase
from src.domain.knowledge.errors import KnowledgeBaseNotFoundError
from src.domain.knowledge.repositories import (
    EmbeddingClient,
    KBRepository,
    Retriever,
)
from src.domain.knowledge.value_objects import (
    KBId,
    RetrievalQuery,
    RetrievedChunk,
    Source,
)
from src.domain.shared.tenant import TenantContext

logger = structlog.get_logger(__name__)

# Trim each Source.content to this many chars before handing to the
# LLM. Mirrors the legacy message_handler's wire-size cap (200) — we
# err on the side of larger context (500) now that providers handle
# 1M-token windows.
_SOURCE_CONTENT_PREVIEW_CHARS = 500


class KnowledgeApplicationService:
    """Use cases for tenant-scoped knowledge listings + RAG retrieval."""

    def __init__(
        self,
        *,
        kb_repository: KBRepository,
        retriever: Retriever,
        embedding_client: EmbeddingClient,
    ) -> None:
        self._kb_repo = kb_repository
        self._retriever = retriever
        self._embedder = embedding_client

    # ── listings ──────────────────────────────────────────────────

    async def list_knowledge_bases(self, *, tenant: TenantContext) -> list[KnowledgeBase]:
        return await self._kb_repo.list_for_tenant(tenant=tenant)

    async def read_knowledge_base(
        self,
        *,
        kb_id: str,
        tenant: TenantContext,
    ) -> KnowledgeBase:
        kb = await self._kb_repo.get(KBId(kb_id), tenant=tenant)
        if kb is None or not kb.is_visible_to(tenant):
            raise KnowledgeBaseNotFoundError(f"KB not found: {kb_id}")
        return kb

    # ── retrieval ─────────────────────────────────────────────────

    async def retrieve_chunks(
        self,
        *,
        tenant: TenantContext,
        query: str,
        kb_ids: tuple[KBId, ...] | None = None,
        top_k: int = 5,
    ) -> list[Source]:
        """Retrieve the most relevant chunks for ``query`` and convert
        them to :class:`Source` view-models ready for LLM injection.

        ``kb_ids=None`` ⇒ search every KB the tenant owns.

        Audit log emits ``mcp.retrieval_*`` events for SOC2 CC7.2 +
        cost telemetry (we log embed token count via the embedder
        adapter's ``dimension`` x N chars heuristic; precise tokens
        require a provider tokeniser which we'll add when the
        embedding adapter exposes usage)."""
        if kb_ids is None:
            visible = await self._kb_repo.list_for_tenant(tenant=tenant)
            kb_ids = tuple(kb.id for kb in visible)

        if not kb_ids:
            return []

        started = time.monotonic()
        logger.info(
            "mcp.retrieval_start",
            tenant_id=tenant.tenant_id,
            kb_count=len(kb_ids),
            top_k=top_k,
            query_len=len(query or ""),
        )

        # Embedding happens inside the retriever for hybrid impls —
        # we pass the raw query string. (Dense-only retrievers would
        # call ``self._embedder.embed([query])`` here and pass the
        # vector to VectorStore.similarity_search directly.)
        retrieval_query = RetrievalQuery(
            query=query,
            kb_ids=kb_ids,
            top_k=top_k,
        )
        retrieved = await self._retriever.retrieve(
            retrieval_query,
            tenant=tenant,
        )

        # Materialize Source view-models. The KB-id → name lookup is
        # cached per-call; for high-throughput we'd add an
        # in-process LRU around list_for_tenant.
        kbs = await self._kb_repo.list_for_tenant(tenant=tenant)
        kb_name_by_id = {kb.id: kb.name for kb in kbs}

        sources: list[Source] = []
        for rc in retrieved:
            kb_name = kb_name_by_id.get(rc.chunk.kb_id) or rc.chunk.kb_id
            sources.append(_to_source(rc, kb_name=str(kb_name)))

        duration_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "mcp.retrieval_ok",
            tenant_id=tenant.tenant_id,
            duration_ms=duration_ms,
            sources=len(sources),
        )
        return sources


# ── helpers ───────────────────────────────────────────────────────


def _to_source(rc: RetrievedChunk, *, kb_name: str) -> Source:
    """Trim chunk content to the preview cap so wire size stays bounded."""
    excerpt = rc.chunk.content or ""
    if len(excerpt) > _SOURCE_CONTENT_PREVIEW_CHARS:
        excerpt = excerpt[: _SOURCE_CONTENT_PREVIEW_CHARS - 1] + "…"
    return Source(
        kb_id=rc.chunk.kb_id,
        kb_name=kb_name,  # type: ignore[arg-type] — str matches KBName NewType
        document_id=rc.chunk.document_id,
        document_title=str(rc.chunk.metadata.get("document_title") or ""),
        chunk_id=rc.chunk.id,
        content=excerpt,
        score=round(rc.score, 4),
        metadata=dict(rc.chunk.metadata or {}),
    )
