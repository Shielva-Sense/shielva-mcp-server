"""
KB Router — routes a query to the most relevant subset of knowledge bases.

Instead of searching all KBs equally for every query, the router embeds the
query and each KB's name/description, computes cosine similarity, and returns
only the top-N most relevant KB IDs.  This is the "query routing" principle
from Vectorless RAG: direct the query to the right index before retrieval.

Falls back to all KBs if routing fails or only 1 KB is configured.
"""
from __future__ import annotations

import math
from typing import List, Dict, Any

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

    async def route(self, query: str, kb_configs: List[Dict[str, Any]]) -> List[str]:
        """
        Return the IDs of the top-N KBs most relevant to `query`.

        Args:
            query: The user's query text
            kb_configs: List of KB config dicts, each with at least an "id" key.
                        "name" and "description" fields improve routing accuracy.

        Returns:
            List of KB IDs (at most self.top_kbs entries, at least 1).
        """
        if len(kb_configs) <= 1:
            return [kb["id"] for kb in kb_configs if "id" in kb]

        # Build a short text descriptor for each KB
        descriptors = []
        for kb in kb_configs:
            name = kb.get("name", "")
            desc = kb.get("description", "")
            descriptors.append(f"{name}. {desc}".strip(". ") or name or kb.get("id", ""))

        # Embed query + all descriptors in one batched call
        texts_to_embed = [query] + descriptors
        try:
            embeddings = await self.embedding_client.embed_batch(texts_to_embed)
        except Exception as exc:
            logger.warning("KBRouter.embed_batch failed, using all KBs", error=str(exc))
            return [kb["id"] for kb in kb_configs if "id" in kb]

        if not embeddings or len(embeddings) != len(texts_to_embed):
            logger.warning("KBRouter got unexpected embedding count, using all KBs")
            return [kb["id"] for kb in kb_configs if "id" in kb]

        query_vec = embeddings[0]
        kb_vecs = embeddings[1:]

        # Score each KB
        scored = []
        for kb, vec in zip(kb_configs, kb_vecs):
            score = _cosine(query_vec, vec)
            scored.append((score, kb.get("id", "")))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [kb_id for _, kb_id in scored[: self.top_kbs] if kb_id]

        logger.info(
            "KBRouter scored KBs",
            query_preview=query[:80],
            scores=[(round(s, 4), kid) for s, kid in scored],
            selected=top,
        )

        return top if top else [kb["id"] for kb in kb_configs if "id" in kb]
