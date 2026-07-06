"""Retrieval adapters — implement the knowledge-context ports
backed by the existing ``rag-engine`` symlink + Gemini embedder.

Slice 3 ships wrappers; slice 4 (or later) will move the underlying
implementations into ``infrastructure/retrieval/`` proper and
delete the legacy import paths.
"""
from .supabase_vector_store_adapter import (
    PgVectorStoreAdapter,
    SupabaseVectorStoreAdapter,
)
from .embedding_client_adapter import LegacyEmbeddingClientAdapter
from .hybrid_retriever_adapter import LegacyHybridRetrieverAdapter

__all__ = [
    "PgVectorStoreAdapter",
    "SupabaseVectorStoreAdapter",  # backward-compat alias
    "LegacyEmbeddingClientAdapter",
    "LegacyHybridRetrieverAdapter",
]
