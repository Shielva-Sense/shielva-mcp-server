"""Knowledge-context ports.

Three storage-adjacent ports + one retrieval-strategy port. Adapters
may implement multiple ports (Supabase backs both VectorStore and
EmbeddingClient via its own pipeline) but the domain contracts are
independent.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from ..shared.tenant import TenantContext
from .entities import KnowledgeBase
from .value_objects import (
    Chunk, ChunkId, KBId, RetrievalQuery, RetrievedChunk,
)


# ── KB metadata ───────────────────────────────────────────────────

class KBRepository(ABC):
    """KB CRUD + per-tenant listings. Holds metadata, NOT vectors."""

    @abstractmethod
    async def get(self, kb_id: KBId, *, tenant: TenantContext) -> Optional[KnowledgeBase]:
        """Tenant-scoped lookup. Returns None when KB doesn't exist
        OR belongs to another tenant — never raise to avoid leaking
        existence."""

    @abstractmethod
    async def list_for_tenant(self, *, tenant: TenantContext) -> List[KnowledgeBase]:
        """All KBs the tenant owns."""

    @abstractmethod
    async def list_for_bot(self, *, bot_id: str, tenant: TenantContext
                           ) -> List[KnowledgeBase]:
        """KBs bound to a specific bot (subset of list_for_tenant)."""

    @abstractmethod
    async def save(self, kb: KnowledgeBase) -> None:
        """Upsert. Used by ProvisionKB (slice 4)."""


# ── Vector store ──────────────────────────────────────────────────

class VectorStore(ABC):
    """Backs similarity search. The infrastructure adapter knows
    where the vectors live (pgvector, Pinecone…)."""

    @abstractmethod
    async def upsert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        *,
        tenant: TenantContext,
    ) -> int:
        """Index chunks. Returns count actually persisted (provider
        may dedupe by chunk id)."""

    @abstractmethod
    async def similarity_search(
        self,
        embedding: List[float],
        *,
        kb_ids: Tuple[KBId, ...],
        tenant: TenantContext,
        top_k:  int = 5,
    ) -> List[RetrievedChunk]:
        """Dense retrieval. ``kb_ids`` filters by KB; tenant is the
        primary isolation key (vectors live in a tenant-scoped
        namespace per the spec)."""

    @abstractmethod
    async def keyword_search(
        self,
        query: str,
        *,
        kb_ids: Tuple[KBId, ...],
        tenant: TenantContext,
        top_k:  int = 5,
    ) -> List[RetrievedChunk]:
        """BM25-style lexical search. Used by hybrid retrievers."""

    @abstractmethod
    async def delete_chunks(
        self,
        chunk_ids: List[ChunkId],
        *,
        tenant: TenantContext,
    ) -> int:
        """Return count actually deleted."""


# ── Embedding client ──────────────────────────────────────────────

class EmbeddingClient(ABC):
    """Turns text into vectors. Adapters carry their own batching."""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Returns one vector per text, in the same order. Provider
        rate-limit / retry logic lives in the adapter."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension — needed by VectorStore.create_collection."""

    @property
    @abstractmethod
    def model(self) -> str:
        """For logging + audit."""


# ── Retrieval strategy ────────────────────────────────────────────

class Retriever(ABC):
    """Composes VectorStore + EmbeddingClient (+ optional reranker)
    into a single ``retrieve`` call.

    Why a separate port rather than baking the strategy into the
    application service:
        * Multiple strategies coexist (dense-only for short
          factoid queries, hybrid for long-form, BM25-only when
          embeddings are stale). Choice is configured per bot, not
          per request — the application service holds a single
          ``Retriever`` and lets the adapter decide.
        * Tests use a fake that returns canned chunks.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: RetrievalQuery,
        *,
        tenant: TenantContext,
    ) -> List[RetrievedChunk]:
        ...
