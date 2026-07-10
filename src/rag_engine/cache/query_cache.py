"""
RAG Engine - Query Cache
Caches query results to improve performance and reduce costs.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib
import json
import structlog
from abc import ABC, abstractmethod

logger = structlog.get_logger(__name__)


@dataclass
class CachedResult:
    """Cached query result"""
    query_hash: str
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    cached_at: datetime
    ttl_seconds: int
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        expiry = self.cached_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry


class QueryCache(ABC):
    """Abstract base class for query caches."""
    
    @abstractmethod
    async def get(self, query_hash: str) -> Optional[CachedResult]:
        """Get cached result."""
        pass
    
    @abstractmethod
    async def set(
        self,
        query_hash: str,
        results: List[Any],
        metadata: Dict[str, Any],
        ttl_seconds: int
    ) -> None:
        """Set cache entry."""
        pass
    
    @abstractmethod
    async def delete(self, query_hash: str) -> None:
        """Delete cache entry."""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass
    
    @staticmethod
    def hash_query(
        query: str,
        tenant_id: str,
        kb_ids: List[str],
        top_k: int = 10
    ) -> str:
        """
        Generate hash for query cache key.
        
        Args:
            query: Search query
            tenant_id: Tenant identifier
            kb_ids: Knowledge base IDs
            top_k: Number of results
            
        Returns:
            Hash string
        """
        cache_key = {
            "query": query.lower().strip(),
            "tenant_id": tenant_id,
            "kb_ids": sorted(kb_ids),
            "top_k": top_k
        }
        key_str = json.dumps(cache_key, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()


class InMemoryCache(QueryCache):
    """
    In-memory cache using Python dict.
    
    Fast but not persistent. Good for development or single-instance deployments.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of entries
        """
        self.max_size = max_size
        self._cache: Dict[str, CachedResult] = {}
        self._access_order: List[str] = []
        
        logger.info("InMemoryCache initialized", max_size=max_size)
    
    async def get(self, query_hash: str) -> Optional[CachedResult]:
        """Get cached result."""
        if query_hash in self._cache:
            result = self._cache[query_hash]
            
            # Check expiry
            if result.is_expired():
                await self.delete(query_hash)
                logger.debug("Cache entry expired", query_hash=query_hash)
                return None
            
            # Update access order (LRU)
            if query_hash in self._access_order:
                self._access_order.remove(query_hash)
            self._access_order.append(query_hash)
            
            logger.debug("Cache hit", query_hash=query_hash)
            return result
        
        logger.debug("Cache miss", query_hash=query_hash)
        return None
    
    async def set(
        self,
        query_hash: str,
        results: List[Any],
        metadata: Dict[str, Any],
        ttl_seconds: int = 3600
    ) -> None:
        """Set cache entry."""
        # Evict if at max size
        if len(self._cache) >= self.max_size and query_hash not in self._cache:
            # Remove least recently used
            if self._access_order:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
                logger.debug("Evicted LRU entry", query_hash=lru_key)
        
        # Serialize results
        serialized_results = [
            asdict(r) if hasattr(r, '__dataclass_fields__') else r
            for r in results
        ]
        
        # Create cache entry
        cached = CachedResult(
            query_hash=query_hash,
            results=serialized_results,
            metadata=metadata,
            cached_at=datetime.utcnow(),
            ttl_seconds=ttl_seconds
        )
        
        self._cache[query_hash] = cached
        
        # Update access order
        if query_hash in self._access_order:
            self._access_order.remove(query_hash)
        self._access_order.append(query_hash)
        
        logger.debug(
            "Cache entry set",
            query_hash=query_hash,
            num_results=len(results),
            ttl=ttl_seconds
        )
    
    async def delete(self, query_hash: str) -> None:
        """Delete cache entry."""
        if query_hash in self._cache:
            del self._cache[query_hash]
            if query_hash in self._access_order:
                self._access_order.remove(query_hash)
            logger.debug("Cache entry deleted", query_hash=query_hash)
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        self._access_order.clear()
        logger.info("Cache cleared", entries_removed=count)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "utilization": len(self._cache) / self.max_size if self.max_size > 0 else 0
        }


class RedisCache(QueryCache):
    """
    Redis-based cache.
    
    Persistent and supports distributed deployments.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "shielva:query_cache:",
        default_ttl: int = 3600
    ):
        """
        Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for cache keys
            default_ttl: Default TTL in seconds
        """
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.redis = None
        
        logger.info("RedisCache initialized", redis_url=redis_url)
    
    async def _get_redis(self):
        """Lazy load Redis client."""
        if self.redis is None:
            try:
                import redis.asyncio as redis
                self.redis = await redis.from_url(self.redis_url, decode_responses=True)
                logger.info("Redis client connected")
            except ImportError:
                logger.error("redis package not installed")
                raise
            except Exception as e:
                logger.error("Failed to connect to Redis", error=str(e))
                raise
        return self.redis
    
    def _make_key(self, query_hash: str) -> str:
        """Make Redis key."""
        return f"{self.key_prefix}{query_hash}"
    
    async def get(self, query_hash: str) -> Optional[CachedResult]:
        """Get cached result from Redis."""
        try:
            redis = await self._get_redis()
            key = self._make_key(query_hash)
            
            data = await redis.get(key)
            if data:
                cached_data = json.loads(data)
                
                result = CachedResult(
                    query_hash=cached_data["query_hash"],
                    results=cached_data["results"],
                    metadata=cached_data["metadata"],
                    cached_at=datetime.fromisoformat(cached_data["cached_at"]),
                    ttl_seconds=cached_data["ttl_seconds"]
                )
                
                logger.debug("Redis cache hit", query_hash=query_hash)
                return result
            
            logger.debug("Redis cache miss", query_hash=query_hash)
            return None
            
        except Exception as e:
            logger.error("Redis get failed", error=str(e))
            return None
    
    async def set(
        self,
        query_hash: str,
        results: List[Any],
        metadata: Dict[str, Any],
        ttl_seconds: int = None
    ) -> None:
        """Set cache entry in Redis."""
        try:
            redis = await self._get_redis()
            key = self._make_key(query_hash)
            
            ttl = ttl_seconds or self.default_ttl
            
            # Serialize results
            serialized_results = [
                asdict(r) if hasattr(r, '__dataclass_fields__') else r
                for r in results
            ]
            
            cached_data = {
                "query_hash": query_hash,
                "results": serialized_results,
                "metadata": metadata,
                "cached_at": datetime.utcnow().isoformat(),
                "ttl_seconds": ttl
            }
            
            await redis.setex(
                key,
                ttl,
                json.dumps(cached_data)
            )
            
            logger.debug(
                "Redis cache entry set",
                query_hash=query_hash,
                num_results=len(results),
                ttl=ttl
            )
            
        except Exception as e:
            logger.error("Redis set failed", error=str(e))
    
    async def delete(self, query_hash: str) -> None:
        """Delete cache entry from Redis."""
        try:
            redis = await self._get_redis()
            key = self._make_key(query_hash)
            await redis.delete(key)
            logger.debug("Redis cache entry deleted", query_hash=query_hash)
        except Exception as e:
            logger.error("Redis delete failed", error=str(e))
    
    async def clear(self) -> None:
        """Clear all cache entries with prefix."""
        try:
            redis = await self._get_redis()
            pattern = f"{self.key_prefix}*"
            
            cursor = 0
            count = 0
            while True:
                cursor, keys = await redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await redis.delete(*keys)
                    count += len(keys)
                if cursor == 0:
                    break
            
            logger.info("Redis cache cleared", entries_removed=count)
            
        except Exception as e:
            logger.error("Redis clear failed", error=str(e))
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")


class NoOpCache(QueryCache):
    """
    No-op cache that doesn't cache anything.
    Useful for testing or when caching is disabled.
    """
    
    async def get(self, query_hash: str) -> Optional[CachedResult]:
        """Always return None (cache miss)."""
        return None
    
    async def set(
        self,
        query_hash: str,
        results: List[Any],
        metadata: Dict[str, Any],
        ttl_seconds: int = 3600
    ) -> None:
        """Do nothing."""
        pass
    
    async def delete(self, query_hash: str) -> None:
        """Do nothing."""
        pass
    
    async def clear(self) -> None:
        """Do nothing."""
        pass
