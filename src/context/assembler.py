"""
MCP Context Assembler
Assembles context for LLM queries by gathering:
- Bot configuration
- Knowledge base content (via RAG)
- Session memory
- Tool context
"""
import hashlib
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import structlog

from src.protocol.models import (
    TenantContext, SessionContext, MCPMessage, MessageRole
)

logger = structlog.get_logger(__name__)

# Context budget constants
_MAX_CHUNK_CHARS = 600
_MAX_KNOWLEDGE_CHARS = 2500


@dataclass
class AssembledContext:
    """Context assembled for LLM query"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    retrieved_chunks: List[Any] = field(default_factory=list)
    bot_config: Dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    context_tokens: int = 0


class ContextAssembler:
    """
    Assembles context for LLM queries.

    Pipeline:
    1. Load bot configuration
    2. Build system prompt
    3. Retrieve knowledge from RAG (with cache + KB routing)
    4. Load session history
    5. Pack into message format
    """

    def __init__(
        self,
        rag_client,
        bot_registry,
        session_store,
        prompt_engine,
        query_cache=None,
        kb_router=None,
    ):
        self.rag_client = rag_client
        self.bot_registry = bot_registry
        self.session_store = session_store
        self.prompt_engine = prompt_engine
        self.query_cache = query_cache   # P1: RedisCache / InMemoryCache / NoOpCache
        self.kb_router = kb_router       # P3: KBRouter for KB-level query routing

        logger.info("ContextAssembler initialized",
                    cache_enabled=query_cache is not None,
                    kb_routing_enabled=kb_router is not None)

    async def assemble(
        self,
        query: str,
        session: SessionContext,
        tenant_context: TenantContext,
        bot_id: str,
        custom_prompt: str = None
    ) -> AssembledContext:
        """
        Assemble full context for LLM query.

        Args:
            query: User query
            session: Current session context
            tenant_context: Tenant isolation context
            bot_id: Bot identifier
            custom_prompt: Optional in-memory system prompt override from Studio

        Returns:
            AssembledContext ready for LLM
        """
        logger.info(
            "Assembling context",
            tenant_id=tenant_context.tenant_id,
            bot_id=bot_id
        )

        # 1. Load bot configuration
        bot_config = await self.bot_registry.get_bot(
            bot_id=bot_id,
            tenant_id=tenant_context.tenant_id
        )

        # 2. Build system prompt
        system_prompt = await self._build_system_prompt(
            bot_config=bot_config,
            tenant_context=tenant_context,
            custom_prompt=custom_prompt
        )

        # 3. Retrieve knowledge
        retrieved_chunks = await self._retrieve_knowledge(
            query=query,
            bot_config=bot_config,
            tenant_context=tenant_context
        )

        # 4. Build context string from retrieved chunks
        knowledge_context = self._format_knowledge_context(retrieved_chunks)

        # 5. Build message list
        messages = await self._build_messages(
            system_prompt=system_prompt,
            knowledge_context=knowledge_context,
            query=query,
            session=session
        )

        return AssembledContext(
            messages=messages,
            retrieved_chunks=retrieved_chunks,
            bot_config=bot_config,
            system_prompt=system_prompt,
            context_tokens=self._estimate_tokens(messages)
        )

    async def _build_system_prompt(
        self,
        bot_config: Dict[str, Any],
        tenant_context: TenantContext,
        custom_prompt: str = None
    ) -> str:
        """Build system prompt from bot config and templates."""
        if custom_prompt:
            base_prompt = custom_prompt
        else:
            base_prompt = bot_config.get("prompt_config", {}).get(
                "system_prompt",
                "You are a helpful AI assistant."
            )

        from datetime import datetime
        current_time_str = datetime.now().strftime("%A, %d %B %Y at %I:%M %p")

        tenant_prompt = f"""
You are operating for tenant: {tenant_context.tenant_id}
User role: {tenant_context.role}
Current Date & Time: {current_time_str}

Response Guidelines:
1. **Format your entire response as valid HTML.** Do not use markdown (no **bold**, no *italics*, no `code`).
2. **CRITICAL:** Do NOT wrap your response in markdown code blocks (like ```html ... ```). Return raw HTML only.
3. Use `<ul>` and `<li>` for lists of messages or items.
4. Use `<strong>` for bold text (e.g., author names or key terms).
5. Use `<p>` for paragraphs.
6. Do not include `<html>`, `<head>`, or `<body>` tags. Just return the content HTML.
7. Group information by author or source if many items are present.
8. Be conversational and helpful. Start with a direct answer like "<p>Yes, I found some messages from...</p>".
9. Only use information from the provided knowledge base.
10. If you don't know something, say so clearly (wrapped in `<p>`).
11. Cite sources or authors precisely.

**Logic & Reasoning:**
- **Time Check:** Compare any dates mentioned in the knowledge base (e.g., "14th Feb") with the **Current Date & Time ({current_time_str})**.
    - If the meeting date is in the **past**: State clearly that the user had a meeting but it seems they missed it or it is over.
    - If the meeting date is in the **future**: State that they have an upcoming meeting and offer to notify them.
- **Name Resolution:** intelligently map usernames like "vivek.sinha" to "Vivek" or other valid names when referring to people.
"""

        tool_instructions = bot_config.get("prompt_config", {}).get(
            "tool_instructions",
            ""
        )

        return f"{base_prompt}\n\n{tenant_prompt}\n\n{tool_instructions}"

    # ── P1: cache key helper ───────────────────────────────────────────

    @staticmethod
    def _cache_key(query: str, tenant_id: str, kb_ids: List[str], top_k: int) -> str:
        """SHA-256 cache key for a RAG retrieval call."""
        payload = json.dumps(
            {"q": query, "t": tenant_id, "k": sorted(kb_ids), "n": top_k},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    async def _retrieve_knowledge(
        self,
        query: str,
        bot_config: Dict[str, Any],
        tenant_context: TenantContext
    ) -> List[Any]:
        """
        Retrieve relevant knowledge from RAG engine.

        Improvements:
        - P1: Check Redis/in-memory cache first; write back on miss
        - P2: top_k=10, drop results below 30% of top score
        - P3: Route query to most-relevant KB subset when kb_router is available
        """
        kb_ids = bot_config.get("kb_ids", [])

        if not kb_ids:
            logger.warning("No knowledge bases configured for bot")
            return []

        top_k = 10  # P2: increased from 5

        # P3: KB routing — narrow to most relevant KBs when multiple are configured
        effective_kb_ids = kb_ids
        if self.kb_router and len(kb_ids) > 1:
            kb_configs = bot_config.get("kbs", [])
            if kb_configs and isinstance(kb_configs[0], dict):
                try:
                    effective_kb_ids = await self.kb_router.route(query, kb_configs)
                    logger.info(
                        "KB routing applied",
                        original_count=len(kb_ids),
                        routed_count=len(effective_kb_ids),
                        kb_ids=effective_kb_ids,
                    )
                except Exception as route_err:
                    logger.warning("KB routing failed, using all KBs", error=str(route_err))
                    effective_kb_ids = kb_ids

        if not self.rag_client:
            return []

        # P1: Check cache
        cache_key = self._cache_key(query, tenant_context.tenant_id, effective_kb_ids, top_k)
        if self.query_cache:
            try:
                cached = await self.query_cache.get(cache_key)
                if cached is not None:
                    logger.info("RAG cache hit", num_results=len(cached))
                    return cached
            except Exception as cache_err:
                logger.warning("Cache get failed", error=str(cache_err))

        try:
            results = await self.rag_client.retrieve(
                query=query,
                tenant_id=tenant_context.tenant_id,
                kb_ids=effective_kb_ids,
                top_k=top_k,
                rerank=True
            )

            # P2: Score threshold — drop chunks scoring < 30% of top result's score
            if results:
                min_score = results[0].score * 0.30
                before = len(results)
                results = [r for r in results if r.score >= min_score]
                if len(results) < before:
                    logger.info(
                        "Score threshold applied",
                        before=before,
                        after=len(results),
                        min_score=round(min_score, 4),
                    )

            logger.info(
                "RAG retrieval successful",
                num_results=len(results),
                kb_ids=effective_kb_ids,
            )

            # P1: Write to cache
            if self.query_cache and results:
                try:
                    await self.query_cache.set(cache_key, results)
                except Exception as cache_err:
                    logger.warning("Cache set failed", error=str(cache_err))

            return results

        except Exception as e:
            logger.error("RAG retrieval failed", error=str(e))
            return []

    def _format_knowledge_context(
        self,
        chunks: List[Any]
    ) -> str:
        """
        Format retrieved chunks into context string.

        Improvements:
        - P4: Emit Section heading when present in chunk metadata
        - P5: Sort by score desc; cap each chunk at 600 chars; total budget 2500 chars
        """
        if not chunks:
            return "No relevant knowledge found."

        # P5: Sort by score descending so best chunks get injected first
        sorted_chunks = sorted(chunks, key=lambda c: getattr(c, "score", 0), reverse=True)

        context_parts = ["<knowledge_base>"]
        total_chars = 0

        for i, chunk in enumerate(sorted_chunks, 1):
            # P5: Budget check — stop once we've hit the total limit
            if total_chars >= _MAX_KNOWLEDGE_CHARS:
                logger.debug("Knowledge context budget reached", chunks_used=i - 1)
                break

            metadata = chunk.metadata or {}
            source = metadata.get("source", "Unknown")
            author = metadata.get("author", "Unknown")
            score = chunk.score

            # P5: Cap individual chunk content
            content = chunk.content
            if len(content) > _MAX_CHUNK_CHARS:
                content = content[:_MAX_CHUNK_CHARS] + "…"

            # P4: Section heading
            section_line = ""
            section_heading = metadata.get("section_heading") or metadata.get("parent_section")
            if section_heading:
                section_line = f"\nSection: {section_heading}"

            block = (
                f"\n<source_{i}>"
                f"\nSource: {source}"
                f"\nAuthor: {author}"
                f"{section_line}"
                f"\nRelevance: {score:.2f}"
                f"\nContent: {content}"
                f"\n</source_{i}>\n"
            )
            context_parts.append(block)
            total_chars += len(content)

        context_parts.append("</knowledge_base>")
        return "\n".join(context_parts)

    async def _build_messages(
        self,
        system_prompt: str,
        knowledge_context: str,
        query: str,
        session: SessionContext
    ) -> List[Dict[str, str]]:
        """Build message list for LLM."""
        messages = []

        full_system = f"""{system_prompt}

<context>
{knowledge_context}
</context>

Answer the user's question based on the context above. If the answer is not in the context, say so.
"""
        messages.append({
            "role": "system",
            "content": full_system
        })

        max_history = 10
        history = session.messages[-max_history:]

        for msg in history:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })

        messages.append({
            "role": "user",
            "content": query
        })

        return messages

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token count (4 chars ≈ 1 token)."""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return total_chars // 4
