"""LLM application service — wraps the LLMProvider port with audit."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

import structlog

from src.domain.llm.repositories import LLMProvider
from src.domain.llm.value_objects import LLMRequest, LLMResponse, LLMStreamChunk
from src.domain.shared.tenant import TenantContext

logger = structlog.get_logger(__name__)


class LLMApplicationService:
    """Single-method use case (today). Future siblings — streaming
    completion, tool-loop orchestration — land in slice 4."""

    def __init__(self, *, provider: LLMProvider) -> None:
        self._provider = provider

    async def complete(
        self,
        request: LLMRequest,
        *,
        tenant: TenantContext,
    ) -> LLMResponse:
        started = time.monotonic()
        logger.info(
            "mcp.llm_complete_start",
            tenant_id=tenant.tenant_id,
            model=str(request.model) if request.model else "default",
            msg_count=len(request.messages),
            has_tools=bool(request.tools),
            tool_count=len(request.tools or ()),
            max_tokens=request.max_tokens,
        )
        try:
            response = await self._provider.complete(request, tenant=tenant)
        except Exception as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                "mcp.llm_complete_failed",
                tenant_id=tenant.tenant_id,
                duration_ms=duration_ms,
                error=str(exc)[:200],
            )
            raise

        duration_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "mcp.llm_complete_ok",
            tenant_id=tenant.tenant_id,
            model=str(response.model),
            finish_reason=response.finish_reason.value,
            tokens_used=response.usage.total_tokens,
            tool_calls_emitted=len(response.tool_calls),
            duration_ms=duration_ms,
        )
        return response

    async def stream(
        self,
        request: LLMRequest,
        *,
        tenant: TenantContext,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Token-by-token streaming completion. Yields deltas as they arrive."""
        started = time.monotonic()
        logger.info(
            "mcp.llm_stream_start",
            tenant_id=tenant.tenant_id,
            model=str(request.model) if request.model else "default",
            msg_count=len(request.messages),
        )
        chunks = 0
        try:
            async for chunk in self._provider.stream(request, tenant=tenant):
                chunks += 1
                yield chunk
        except Exception as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                "mcp.llm_stream_failed",
                tenant_id=tenant.tenant_id,
                duration_ms=duration_ms,
                chunks=chunks,
                error=str(exc)[:200],
            )
            raise
        duration_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "mcp.llm_stream_ok",
            tenant_id=tenant.tenant_id,
            chunks=chunks,
            duration_ms=duration_ms,
        )
