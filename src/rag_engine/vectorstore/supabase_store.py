"""
RAG Engine - Vector Store Integration with pgvector (vecs)
Multi-tenant vector storage on plain in-cluster Postgres + pgvector via vecs.
(Formerly named "Supabase"; the DB is plain pgvector, not Supabase.)
"""

import asyncio
from typing import Any

import structlog
import vecs

logger = structlog.get_logger(__name__)


from .models import SearchResult, VectorDocument


class PgVectorStore:
    """
    pgvector vector store using vecs (plain in-cluster Postgres + pgvector).

    Features:
    - Collection management via vecs
    - Metadata filtering
    - Dot product / Cosine distance
    """

    def __init__(
        self,
        db_url: str,
        collection_prefix: str = "shielva_kb_",
        embedding_dim: int = 768,  # Default for Gemini
    ):
        """
        Initialize pgvector vector store.

        Args:
            db_url: PostgreSQL connection string
            collection_prefix: Prefix for collection names
            embedding_dim: Embedding dimension
        """
        self.db_url = db_url
        self.collection_prefix = collection_prefix
        self.embedding_dim = embedding_dim

        self._client: vecs.Client | None = None

        logger.info("PgVectorStore initialized")

    async def connect(self):
        """Establish connection to pgvector (via vecs)."""
        # vecs.create_client is synchronous but lightweight (connection pool)
        try:
            self._client = vecs.create_client(self.db_url)
            logger.info("Connected to pgvector")
        except Exception as e:
            logger.error("Failed to connect to pgvector", error=str(e))
            raise e

    async def close(self):
        """Close connection."""
        if self._client:
            self._client.disconnect()

    def _get_collection_name(self, tenant_id: str, kb_id: str) -> str:
        """Get collection name for tenant and KB with hashing for long identifiers."""
        # Combine checks
        raw_name = f"{self.collection_prefix}{tenant_id}_{kb_id}"

        # PostgreSQL identifier limit is 63 bytes
        if len(raw_name) <= 63:
            return raw_name.replace("-", "_")

        # If too long, hash the unique components
        import hashlib

        # Keep prefix for readability (11 chars)
        # Hash tenant+kb for uniqueness (32 chars hex)
        # Total = 11 + 32 = 43 chars < 63
        unique_str = f"{tenant_id}_{kb_id}"
        hash_suffix = hashlib.sha256(unique_str.encode()).hexdigest()[:32]
        return f"{self.collection_prefix}{hash_suffix}"

    async def create_collection(self, tenant_id: str, kb_id: str) -> str:
        """Create a new collection."""
        collection_name = self._get_collection_name(tenant_id, kb_id)

        # vecs creates the table/index if it doesn't exist
        # We perform this in a sync way as vecs is sync currently
        try:
            self._client.get_or_create_collection(name=collection_name, dimension=self.embedding_dim)
            logger.info("Created/Retrieved collection", collection=collection_name)

            # Ensure text index exists
            # We call this async method. But create_collection is async so it's fine.
            # But wait, create_text_index uses self._client.Session() which is synchronous sqlalchemy session?
            # vecs client is sync wrapper around sqlalchemy.
            # But my method signatures are async.
            # I should await it.
            await self.create_text_index(tenant_id, kb_id)

            return collection_name
        except Exception as e:
            logger.error("Failed to create collection", error=str(e))
            raise e

    async def delete_collection(self, tenant_id: str, kb_id: str) -> bool:
        """Delete a collection."""
        collection_name = self._get_collection_name(tenant_id, kb_id)

        try:
            self._client.delete_collection(collection_name)
            logger.info("Deleted collection", collection=collection_name)
            return True
        except Exception as e:
            logger.error("Failed to delete collection", error=str(e))
            return False

    async def upsert(self, tenant_id: str, kb_id: str, documents: list[VectorDocument]) -> int:
        """Upsert documents."""
        collection_name = self._get_collection_name(tenant_id, kb_id)

        # Get collection
        collection = self._client.get_or_create_collection(name=collection_name, dimension=self.embedding_dim)

        # Prepare records: list of (id, vector, metadata)
        records = []
        for doc in documents:
            metadata = {"content": doc.content, "tenant_id": tenant_id, "kb_id": kb_id, **doc.metadata}
            records.append((doc.id, doc.embedding, metadata))

        # Batch upsert is handled by vecs? vecs has .upsert(records)
        try:
            collection.upsert(records=records)
            logger.info("Upserted documents", collection=collection_name, count=len(records))
            return len(records)
        except Exception as e:
            logger.error("Upsert failed", error=str(e))
            raise e

    async def search(
        self,
        tenant_id: str,
        kb_ids: list[str],
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict[str, Any] = None,
    ) -> list[SearchResult]:
        """Search across KBs."""
        all_results: list[SearchResult] = []
        if not kb_ids:
            return all_results

        def _search_one(kb_id: str) -> list[SearchResult]:
            """Blocking per-KB vecs query. Run via asyncio.to_thread so the sync
            SQLAlchemy/vecs round-trip never stalls the event loop (otherwise one
            slow query serialises every concurrent bot answer)."""
            collection_name = self._get_collection_name(tenant_id, kb_id)
            out: list[SearchResult] = []
            try:
                try:
                    collection = self._client.get_collection(name=collection_name)
                except KeyError:
                    return out  # Collection doesn't exist

                # Single query (content lives in metadata). Previously this method
                # ran the same query twice + opened an unused Session — removed.
                results = collection.query(
                    data=query_embedding,
                    limit=top_k,
                    filters=filters,
                    include_metadata=True,
                )
                if not results:
                    return out

                found_ids = [r[0] for r in results] if not isinstance(results[0], str) else results
                if not found_ids:
                    return out

                records = collection.fetch(ids=found_ids)
                record_map = {r.id: r for r in records}

                for rank, local_id in enumerate(found_ids):
                    rec = record_map.get(local_id)
                    if rec is None:
                        continue
                    # Synthetic rank-based score — preserves order for RRF fusion.
                    score = 1.0 / (1.0 + rank)
                    out.append(
                        SearchResult(
                            id=rec.id,
                            content=rec.metadata.get("content", ""),
                            score=score,
                            metadata={
                                k: v for k, v in rec.metadata.items() if k not in ["content", "tenant_id", "kb_id"]
                            },
                        )
                    )
            except Exception as e:
                logger.error("Search failed", collection=collection_name, error=str(e))
            return out

        # One thread per KB, all concurrent and off the event loop.
        per_kb = await asyncio.gather(*[asyncio.to_thread(_search_one, kb_id) for kb_id in kb_ids])
        for sub in per_kb:
            all_results.extend(sub)

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    async def create_text_index(self, tenant_id: str, kb_id: str):
        """
        Create a GIN index on metadata->'content' for Full Text Search.
        """
        collection_name = self._get_collection_name(tenant_id, kb_id)
        # vecs maps collection name to table name in 'vecs' schema.
        # usually "vecs"."collection_name"

        # We need to execute raw SQL.
        try:
            from sqlalchemy import text

            with self._client.Session() as sess:
                # 1. Add immutable generated column for tsvector if not exists (optional, but better for perf)
                # For simplicity, we'll index the expression directly.
                # "vecs".{collection_name} is the table.

                # Note: vecs table structure has 'metadata' as JSONB.

                sess.execute(
                    text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{collection_name}_content_fts
                    ON vecs."{collection_name}"
                    USING GIN (to_tsvector('english', metadata->>'content'));
                """)
                )
                # ANN index on the embedding column → sub-linear vector search
                # (HNSW, cosine). Without it pgvector does an O(n) exact scan.
                # vecs stores the vector in the `vec` column. Idempotent.
                sess.execute(
                    text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{collection_name}_vec_hnsw
                    ON vecs."{collection_name}"
                    USING hnsw (vec vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
                )
                sess.commit()
                logger.info("Created FTS + HNSW indexes", collection=collection_name)
        except Exception as e:
            logger.error("Failed to create indexes", error=str(e))
            # Don't raise, just log.

    async def keyword_search(
        self, tenant_id: str, kb_ids: list[str], query: str, top_k: int = 10
    ) -> list[SearchResult]:
        """
        Perform keyword search using Postgres Full Text Search.
        """
        from sqlalchemy import text

        all_results: list[SearchResult] = []
        if not kb_ids:
            return all_results

        def _kw_one(kb_id: str) -> list[SearchResult]:
            """Blocking per-KB Postgres FTS query — run off the event loop."""
            collection_name = self._get_collection_name(tenant_id, kb_id)
            out: list[SearchResult] = []
            try:
                with self._client.Session() as sess:
                    # S608 suppressed below: the only interpolated value is
                    # `collection_name`, a Postgres table identifier (which cannot
                    # be parameter-bound). It is derived internally from tenant_id +
                    # kb_id via `_get_collection_name` (identifier-normalized), never
                    # from user free-text. The user-supplied `query` and `top_k` ARE
                    # bound via :query / :limit below.
                    sql = text(f"""
                        SELECT
                            id,
                            metadata,
                            ts_rank(to_tsvector('english', metadata->>'content'), websearch_to_tsquery('english', :query)) as score
                        FROM vecs."{collection_name}"
                        WHERE to_tsvector('english', metadata->>'content') @@ websearch_to_tsquery('english', :query)
                        ORDER BY score DESC
                        LIMIT :limit;
                    """)  # noqa: S608
                    result = sess.execute(sql, {"query": query, "limit": top_k})
                    for row in result:
                        metadata = row[1] or {}
                        out.append(
                            SearchResult(
                                id=row[0],
                                content=metadata.get("content", ""),
                                score=float(row[2]),  # ts_rank score
                                metadata={
                                    k: v for k, v in metadata.items() if k not in ["content", "tenant_id", "kb_id"]
                                },
                            )
                        )
            except Exception as e:
                # Table might not exist or other error
                logger.warning(
                    "Keyword search failed (maybe no index/table?)", collection=collection_name, error=str(e)
                )
            return out

        # One thread per KB, all concurrent and off the event loop.
        per_kb = await asyncio.gather(*[asyncio.to_thread(_kw_one, kb_id) for kb_id in kb_ids])
        for sub in per_kb:
            all_results.extend(sub)

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    async def delete_by_ids(self, tenant_id: str, kb_id: str, document_ids: list[str]) -> int:
        """Delete by IDs."""
        collection_name = self._get_collection_name(tenant_id, kb_id)
        collection = self._client.get_collection(name=collection_name)
        collection.delete(ids=document_ids)
        return len(document_ids)


# Backward-compat alias: the class was renamed SupabaseVectorStore -> PgVectorStore
# (the DB is plain pgvector, not Supabase). Keep the old name importable for any
# external caller until all references migrate.
SupabaseVectorStore = PgVectorStore
