"""
Vector Store Module
"""
from .models import SearchResult, VectorDocument
from .supabase_store import PgVectorStore, SupabaseVectorStore

__all__ = [
    "PgVectorStore",
    "SearchResult",
    "SupabaseVectorStore",  # backward-compat alias
    "VectorDocument"
]
