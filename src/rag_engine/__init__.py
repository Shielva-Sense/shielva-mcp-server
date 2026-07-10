"""
RAG Engine - Source Code
"""
from .vectorstore import PgVectorStore, SupabaseVectorStore, VectorDocument, SearchResult
from .retriever import HybridRetriever, RetrievalResult
from .reranker import (
    Reranker,
    CrossEncoderReranker,
    CohereReranker,
    LLMReranker,
    NoOpReranker
)
from .cache import (
    QueryCache,
    InMemoryCache,
    RedisCache,
    NoOpCache
)

__all__ = [
    # Vector Store
    "PgVectorStore",
    "SupabaseVectorStore",  # backward-compat alias
    "VectorDocument",
    "SearchResult",
    
    # Retriever
    "HybridRetriever",
    "RetrievalResult",
    
    # Reranker
    "Reranker",
    "CrossEncoderReranker",
    "CohereReranker",
    "LLMReranker",
    "NoOpReranker",
    
    # Cache
    "QueryCache",
    "InMemoryCache",
    "RedisCache",
    "NoOpCache"
]
