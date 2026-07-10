"""
RAG Engine - Retriever with Hybrid Search
Combines vector search with BM25 for better retrieval
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import asyncio
import structlog
import numpy as np

from ..reranker import Reranker
from ..vectorstore import SearchResult

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval with sources"""
    id: str
    content: str
    score: float
    kb_id: str
    kb_name: str
    document_id: str
    document_title: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    """
    Hybrid retriever combining vector search and BM25.
    
    Features:
    - Vector similarity search (semantic)
    - BM25 keyword search (lexical)
    - Score fusion (RRF)
    - Reranking
    """
    
    def __init__(
        self,
        vector_store: Any,
        embedding_client,
        reranker: Optional[Reranker] = None,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        fusion_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Qdrant vector store
            embedding_client: Client to generate embeddings
            reranker: Optional reranker for result quality
            vector_weight: Weight for vector search scores
            bm25_weight: Weight for BM25 scores
            fusion_k: RRF k parameter
        """
        self.vector_store = vector_store
        self.embedding_client = embedding_client
        self.reranker = reranker
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.fusion_k = fusion_k
        
        
        logger.info("HybridRetriever initialized", 
                    has_rrf=hasattr(self, "_reciprocal_rank_fusion"),
                    id=id(self))
        if hasattr(self, "_reciprocal_rank_fusion"):
            logger.info("Method _reciprocal_rank_fusion FOUND")
        else:
            logger.error("Method _reciprocal_rank_fusion MISSING")
    
    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        kb_ids: List[str],
        top_k: int = 10,
        use_hybrid: bool = True,
        rerank: bool = True,
        query_embedding: Optional[List[float]] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query
            tenant_id: Tenant identifier
            kb_ids: Knowledge base IDs to search
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid search
            rerank: Whether to rerank results
            query_embedding: Optional precomputed embedding of `query`. When
                supplied (FIX #5 — embed the query once per turn upstream and
                reuse it), we skip the redundant embedding round-trip. Falls
                back to embedding `query` here when omitted, so existing
                callers are unaffected.

        Returns:
            List of retrieval results
        """
        logger.info(
            "Retrieving documents",
            tenant_id=tenant_id,
            kb_ids=kb_ids,
            query_length=len(query)
        )

        # Get query embedding (reuse the caller's vector when provided)
        if query_embedding is None:
            query_embedding = await self._get_embedding(query)

        use_keyword = use_hybrid and self.bm25_weight > 0
        if use_keyword:
            # Vector search and keyword (FTS) search are independent — run them
            # concurrently instead of one-after-another. With the store's blocking
            # calls offloaded to threads, this genuinely halves the DB-bound time.
            vector_results, bm25_results = await asyncio.gather(
                self.vector_store.search(
                    tenant_id=tenant_id,
                    kb_ids=kb_ids,
                    query_embedding=query_embedding,
                    top_k=top_k * 2,  # Get more for fusion
                ),
                self._keyword_search(
                    query=query,
                    tenant_id=tenant_id,
                    kb_ids=kb_ids,
                    top_k=top_k * 2,
                ),
            )
            # Fuse results using RRF
            results = self._reciprocal_rank_fusion(
                vector_results=vector_results,
                bm25_results=bm25_results,
                top_k=top_k
            )
        else:
            vector_results = await self.vector_store.search(
                tenant_id=tenant_id,
                kb_ids=kb_ids,
                query_embedding=query_embedding,
                top_k=top_k * 2,  # Get more for fusion
            )
            results = vector_results
        
        # Rerank if enabled
        if rerank and self.reranker:
            results = await self.reranker.rerank(
                query=query,
                results=results,
                top_k=top_k
            )
        else:
            results = results[:top_k]
        
        # Convert to RetrievalResult and Deduplicate by content
        seen_content = set()
        unique_results = []
        
        for r in results:
            # Normalize content for comparison
            normalized_content = " ".join(r.content.split()).lower()
            if normalized_content not in seen_content:
                seen_content.add(normalized_content)
                unique_results.append(
                    RetrievalResult(
                        id=r.id,
                        content=r.content,
                        score=r.score,
                        kb_id=r.metadata.get("kb_id", ""),
                        kb_name=r.metadata.get("kb_name", ""),
                        document_id=r.metadata.get("document_id", ""),
                        document_title=r.metadata.get("document_title", ""),
                        chunk_id=r.metadata.get("chunk_id", r.id),
                        metadata=r.metadata
                    )
                )
        
        return unique_results[:top_k]
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single query string.

        The injected EmbeddingClient.embed() takes a LIST and returns a list of
        vectors. Passing a bare string made it iterate the string into
        batch-sized substrings, embed each, and the multiple 768-d vectors got
        flattened into one (e.g. 2×768 = 1536) that then mismatched the 768-d
        pgvector column ("expected 768 dimensions, not 1536") and silently
        returned zero hits. Always request exactly one embedding.
        """
        if self.embedding_client:
            if hasattr(self.embedding_client, "embed_single"):
                return await self.embedding_client.embed_single(text)
            vecs = await self.embedding_client.embed([text])
            return vecs[0] if vecs else [0.0] * 768

        # No embedding client configured — return a dimension-correct zero vector
        # (768 = Gemini models/gemini-embedding-001) so we never mismatch the
        # pgvector column (was [0.0] * 1536, an OpenAI-era leftover that would
        # break distance computation against 768-dim collections). A zero vector
        # yields no false matches; the keyword/BM25 leg still runs.
        logger.warning("No embedding client configured — query falls back to keyword search only")
        return [0.0] * 768
    
    async def _keyword_search(
        self,
        query: str,
        tenant_id: str,
        kb_ids: List[str],
        top_k: int
    ) -> List[SearchResult]:
        """Perform keyword search via vector store (Postgres FTS)."""
        if hasattr(self.vector_store, "keyword_search"):
            return await self.vector_store.keyword_search(
                tenant_id=tenant_id,
                kb_ids=kb_ids,
                query=query,
                top_k=top_k
            )
        else:
            logger.warning("Vector store does not support keyword search")
            return []

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        top_k: int,
        k: int = 60
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from keyword search
            top_k: Number of results to return
            k: RRF constant (default 60)
            
        Returns:
            Fused list of search results
        """
        # Map IDs to results and scores
        results_map: Dict[str, SearchResult] = {}
        scores_map: Dict[str, float] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            if result.id not in results_map:
                results_map[result.id] = result
                scores_map[result.id] = 0.0
            
            # Application of vector weight
            scores_map[result.id] += (1 / (k + rank + 1)) * self.vector_weight
            
        # Process BM25 results
        for rank, result in enumerate(bm25_results):
            if result.id not in results_map:
                results_map[result.id] = result
                scores_map[result.id] = 0.0
                
            # Application of BM25 weight
            scores_map[result.id] += (1 / (k + rank + 1)) * self.bm25_weight
            
        # Sort by fused score
        sorted_ids = sorted(
            scores_map.keys(),
            key=lambda x: scores_map[x],
            reverse=True
        )
        
        # Return top_k results with updated scores
        fused_results = []
        for doc_id in sorted_ids[:top_k]:
            result = results_map[doc_id]
            # Update score with fused score
            result.score = scores_map[doc_id]
            fused_results.append(result)
            
        return fused_results
