"""
RAG Engine - Reranker Module
Reranks search results for improved relevance using cross-encoder models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RerankResult:
    """Result after reranking"""

    id: str
    content: str
    score: float
    original_score: float
    metadata: dict[str, Any]


class Reranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    async def rerank(self, query: str, results: list[Any], top_k: int = 10) -> list[Any]:
        """
        Rerank search results.

        Args:
            query: Search query
            results: List of search results
            top_k: Number of top results to return

        Returns:
            Reranked results
        """


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder based reranker.

    Uses a cross-encoder model to score query-document pairs.
    More accurate than bi-encoder but slower.
    """

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32, device: str = "cpu"
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for inference
            device: Device to run on (cpu/cuda)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None

        logger.info("CrossEncoderReranker initialized", model=model_name, device=device)

    def _load_model(self):
        """Lazy load the model."""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(self.model_name, device=self.device)
                logger.info("Cross-encoder model loaded", model=self.model_name)
            except ImportError:
                logger.warning("sentence-transformers not installed, using fallback reranker")
                self.model = "fallback"

    async def rerank(self, query: str, results: list[Any], top_k: int = 10) -> list[Any]:
        """
        Rerank results using cross-encoder.

        Args:
            query: Search query
            results: List of search results with 'content' attribute
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        self._load_model()

        # Fallback if model not available
        if self.model == "fallback":
            return results[:top_k]

        try:
            # Prepare query-document pairs
            pairs = [[query, r.content] for r in results]

            # Get cross-encoder scores
            scores = self.model.predict(pairs, batch_size=self.batch_size)

            # Combine results with new scores
            reranked = []
            for result, score in zip(results, scores, strict=False):
                # Update the result with reranked score
                result.score = float(score)
                reranked.append(result)

            # Sort by new score
            reranked.sort(key=lambda x: x.score, reverse=True)

            logger.info("Reranked results", original_count=len(results), returned_count=min(top_k, len(reranked)))

            return reranked[:top_k]

        except Exception as e:
            logger.error("Reranking failed, returning original results", error=str(e))
            return results[:top_k]


class CohereReranker(Reranker):
    """
    Cohere Rerank API based reranker.

    Uses Cohere's rerank endpoint for high-quality reranking.
    """

    def __init__(self, api_key: str, model: str = "rerank-english-v2.0", top_n: int = 10):
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key
            model: Cohere rerank model
            top_n: Number of results to return from API
        """
        self.api_key = api_key
        self.model = model
        self.top_n = top_n
        self.client = None

        logger.info("CohereReranker initialized", model=model)

    def _get_client(self):
        """Lazy load Cohere client."""
        if self.client is None:
            try:
                import cohere

                self.client = cohere.Client(self.api_key)
                logger.info("Cohere client initialized")
            except ImportError:
                logger.warning("cohere package not installed")
                self.client = "fallback"

    async def rerank(self, query: str, results: list[Any], top_k: int = 10) -> list[Any]:
        """
        Rerank using Cohere API.

        Args:
            query: Search query
            results: List of search results
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        self._get_client()

        if self.client == "fallback":
            return results[:top_k]

        try:
            # Prepare documents
            documents = [r.content for r in results]

            # Call Cohere rerank
            response = self.client.rerank(
                query=query, documents=documents, model=self.model, top_n=min(self.top_n, len(documents))
            )

            # Map back to original results
            reranked = []
            for item in response.results:
                idx = item.index
                result = results[idx]
                result.score = item.relevance_score
                reranked.append(result)

            logger.info("Cohere reranked results", original_count=len(results), returned_count=len(reranked))

            return reranked[:top_k]

        except Exception as e:
            logger.error("Cohere reranking failed", error=str(e))
            return results[:top_k]


class LLMReranker(Reranker):
    """
    LLM-based reranker using prompt engineering.

    Uses an LLM to judge relevance of documents to query.
    Slower but can provide reasoning.
    """

    def __init__(self, llm_client, model: str = "gpt-3.5-turbo", batch_size: int = 5):
        """
        Initialize LLM reranker.

        Args:
            llm_client: LLM client (e.g., OpenAI, Anthropic)
            model: Model name
            batch_size: Number of docs to score per LLM call
        """
        self.llm_client = llm_client
        self.model = model
        self.batch_size = batch_size

        logger.info("LLMReranker initialized", model=model)

    async def rerank(self, query: str, results: list[Any], top_k: int = 10) -> list[Any]:
        """
        Rerank using LLM scoring.

        Args:
            query: Search query
            results: List of search results
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        if not results:
            return []

        try:
            scored_results = []

            # Process in batches
            for i in range(0, len(results), self.batch_size):
                batch = results[i : i + self.batch_size]

                # Create prompt
                docs_text = "\n\n".join([f"Document {j + 1}:\n{r.content[:500]}" for j, r in enumerate(batch)])

                prompt = f"""Given the query and documents below, rate each document's relevance to the query on a scale of 0-10.

Query: {query}

{docs_text}

Respond with only a JSON array of scores, e.g., [8, 3, 9, 1, 6]"""

                # Get LLM scores
                response = await self.llm_client.complete(prompt=prompt, model=self.model, temperature=0)

                # Parse scores
                import json

                try:
                    scores = json.loads(response)
                    for result, score in zip(batch, scores, strict=False):
                        result.score = float(score) / 10.0  # Normalize to 0-1
                        scored_results.append(result)
                except json.JSONDecodeError:
                    # Fallback to original scores
                    scored_results.extend(batch)

            # Sort by score
            scored_results.sort(key=lambda x: x.score, reverse=True)

            logger.info(
                "LLM reranked results", original_count=len(results), returned_count=min(top_k, len(scored_results))
            )

            return scored_results[:top_k]

        except Exception as e:
            logger.error("LLM reranking failed", error=str(e))
            return results[:top_k]


class NoOpReranker(Reranker):
    """
    No-op reranker that returns results as-is.
    Useful for testing or when reranking is disabled.
    """

    async def rerank(self, query: str, results: list[Any], top_k: int = 10) -> list[Any]:
        """Return results without reranking."""
        return results[:top_k]
