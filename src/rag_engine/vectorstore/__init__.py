"""
Vector Store Module
"""
from .models import VectorDocument, SearchResult
from .supabase_store import PgVectorStore, SupabaseVectorStore

__all__ = [
    "PgVectorStore",
    "SupabaseVectorStore",  # backward-compat alias
    "VectorDocument",
    "SearchResult"
]
