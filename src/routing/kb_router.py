"""
KB Router — routes a query to the most relevant subset of knowledge bases.

Instead of searching all KBs equally for every query, the router embeds the
query and each KB's name/description, computes cosine similarity, and returns
only the top-N most relevant KB IDs.  This is the "query routing" principle
from Vectorless RAG: direct the query to the right index before retrieval.

Falls back to all KBs if routing fails or only 1 KB is configured.
"""
from __future__ import annotations

import hashlib
import math
from typing import List, Dict, Any, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class KBRouter:
    """
    Routes a query to the most relevant KBs using embedding similarity.

    Usage:
        router = KBRouter(embedding_client=client, top_kbs=2)
        kb_ids = await router.route(query, kb_configs)

    kb_configs is the list of KB objects from bot config, e.g.:
        [{"id": "kb-abc", "name": "HR Policy", "description": "..."}, ...]
    """

    def __init__(self, embedding_client, top_kbs: int = 2):
        self.embedding_client = embedding_client
        self.top_kbs = top_kbs
        # In-process cache of KB-descriptor embeddings. KB names/descriptions
        # never change between turns, so re-embedding them every turn is pure
        # waste. Keyed by (kb_id, sha256(descriptor)) so a renamed/edited KB
        # naturally re-embeds while a stable one is reused for the process'
        # lifetime. Embedding vectors are not secrets and are tenant-agnostic
        # at the descriptor level (the kb_id namespaces them).
        self._descriptor_cache: Dict[Tuple[str, str], List[float]] = {}

    @staticmethod
    def _descriptor_for(kb: Dict[str, Any]) -> str:
        """Build the short text descriptor used to embed a KB."""
        name = kb.get("name", "")
        desc = kb.get("description", "")
        return f"{name}. {desc}".strip(". ") or name or kb.get("id", "")

    async def route(
        self,
        query: str,
        kb_configs: List[Dict[str, Any]],
        query_embedding: Optional[List[float]] = None,
    ) -> List[str]:
        """
        Return the IDs of the top-N KBs most relevant to `query`.

        Args:
            query: The user's query text
            kb_configs: List of KB config dicts, each with at least an "id" key.
                        "name" and "description" fields improve routing accuracy.
            query_embedding: Optional precomputed embedding of `query`. When
                        provided (FIX #5 — embed-once-per-turn), the router
                        reuses it instead of re-embedding the query, and only
                        embeds the KB descriptors that aren't already cached.
                        Falls back to embedding the query itself when omitted,
                        so existing callers keep working unchanged.

        Returns:
            List of KB IDs (at most self.top_kbs entries, at least 1).
        """
        kb_ids, _ = await self.route_with_embedding(query, kb_configs, query_embedding)
        return kb_ids

    async def route_with_embedding(
        self,
        query: str,
        kb_configs: List[Dict[str, Any]],
        query_embedding: Optional[List[float]] = None,
    ) -> Tuple[List[str], Optional[List[float]]]:
        """Like :meth:`route` but also returns the query embedding that was
        used (computed here when not supplied). Lets the caller reuse the one
        vector for downstream vector search (FIX #5)."""
        all_ids = [kb["id"] for kb in kb_configs if "id" in kb]

        if len(kb_configs) <= 1:
            return all_ids, query_embedding

        # Resolve the descriptors, splitting into cached vs. needs-embedding.
        descriptors = [self._descriptor_for(kb) for kb in kb_configs]
        cache_keys: List[Tuple[str, str]] = [
            (kb.get("id", ""), hashlib.sha256(d.encode()).hexdigest())
            for kb, d in zip(kb_configs, descriptors)
        ]
        missing_idx = [
            i for i, key in enumerate(cache_keys) if key not in self._descriptor_cache
        ]

        # Build the single batched embed call: the query (only if we don't
        # already have it) + any uncached descriptors.
        texts_to_embed: List[str] = []
        need_query = query_embedding is None
        if need_query:
            texts_to_embed.append(query)
        texts_to_embed.extend(descriptors[i] for i in missing_idx)

        try:
            embeddings = await self.embedding_client.embed(texts_to_embed) if texts_to_embed else []
        except Exception as exc:
            logger.warning("KBRouter.embed failed, using all KBs", error=str(exc))
            return all_ids, query_embedding

        if len(embeddings) != len(texts_to_embed):
            logger.warning("KBRouter got unexpected embedding count, using all KBs")
            return all_ids, query_embedding

        cursor = 0
        if need_query:
            query_embedding = embeddings[cursor]
            cursor += 1
        for i in missing_idx:
            self._descriptor_cache[cache_keys[i]] = embeddings[cursor]
            cursor += 1

        if not query_embedding:
            logger.warning("KBRouter has no query embedding, using all KBs")
            return all_ids, query_embedding

        # Score each KB against the (now fully-cached) descriptor vectors.
        scored = []
        for kb, key in zip(kb_configs, cache_keys):
            vec = self._descriptor_cache.get(key)
            if vec is None:
                continue
            score = _cosine(query_embedding, vec)
            scored.append((score, kb.get("id", "")))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [kb_id for _, kb_id in scored[: self.top_kbs] if kb_id]

        logger.info(
            "KBRouter scored KBs",
            query_preview=query[:80],
            scores=[(round(s, 4), kid) for s, kid in scored],
            selected=top,
        )

        return (top if top else all_ids), query_embedding
