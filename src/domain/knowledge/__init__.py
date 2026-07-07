"""Knowledge bounded context.

Owns the conceptual model of tenant-scoped knowledge: a Knowledge
Base contains Documents, which are split into Chunks, which are
indexed by a VectorStore + retrieved as Sources for the LLM.

Three ports because three different infrastructure stacks live
behind them:
    * :class:`KBRepository`     — Mongo / Postgres (KB metadata).
    * :class:`VectorStore`      — pgvector / Pinecone
                                  (the actual vectors + similarity search).
    * :class:`EmbeddingClient`  — Gemini / OpenAI / Cohere / local model.

Plus one composite read port:
    * :class:`Retriever`        — orchestrates VectorStore + EmbeddingClient
                                  with optional keyword / RRF blending.
                                  Hybrid retrieval is a *strategy*; we model
                                  it as a port so different strategies
                                  (dense-only, BM25-only, hybrid+rerank) are
                                  swappable.
"""

from .entities import KnowledgeBase
from .errors import KnowledgeBaseNotFoundError
from .repositories import (
    EmbeddingClient,
    KBRepository,
    Retriever,
    VectorStore,
)
from .value_objects import (
    Chunk,
    ChunkId,
    Document,
    DocumentId,
    KBId,
    KBName,
    KBStatus,
    RetrievalQuery,
    RetrievedChunk,
    Source,
)

__all__ = [
    "Chunk",
    "ChunkId",
    "Document",
    "DocumentId",
    "EmbeddingClient",
    "KBId",
    "KBName",
    "KBRepository",
    "KBStatus",
    "KnowledgeBase",
    "KnowledgeBaseNotFoundError",
    "RetrievalQuery",
    "RetrievedChunk",
    "Retriever",
    "Source",
    "VectorStore",
]
