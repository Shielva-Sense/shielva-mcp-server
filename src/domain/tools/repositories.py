"""Tools-context ports.

We split read (catalogue) from write (executor) because they have
different latency/observability/auth profiles. Adapters MAY
implement both interfaces on one class — many will — but the domain
contract is two distinct ports.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..shared.tenant import TenantContext
from .entities import Tool
from .value_objects import ToolName, ToolResult


class ToolCatalogue(ABC):
    """Read-side: what tools exist, and what's their schema."""

    @abstractmethod
    async def list_for(self, tenant: TenantContext) -> List[Tool]:
        """Tools available to this tenant.

        Adapters filter by ``Tool.is_permitted_for(tenant)`` so the
        LLM never sees tools it cannot use. Sort order is the
        adapter's choice (typically registration order).
        """

    @abstractmethod
    async def get(self, name: ToolName) -> Optional[Tool]:
        """Single tool by name. None when unknown. Permission check
        is the *executor's* responsibility — this is a pure lookup."""


class ToolExecutor(ABC):
    """Write-side: invoke a tool against the tenant's workspace."""

    @abstractmethod
    async def execute(
        self,
        *,
        tool:      Tool,
        arguments: Dict[str, Any],
        tenant:    TenantContext,
        context:   Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Run the tool. The adapter is responsible for:

        * Validating arguments against ``tool.input_schema`` (some
          providers do this in the LLM, but the executor MUST
          enforce as defence-in-depth).
        * Enforcing ``tool.required_permissions`` — raise
          :class:`ToolPermissionDeniedError` when denied.
        * Catching tool-internal exceptions and converting them to
          ``ToolResult.failure(...)`` (per MCP spec — tool errors
          surface to the LLM, not to the host).

        ``context`` is an opaque kv-bag the interface adapter may
        pass through (e.g. ``mcp_session_id``). Tools that need it
        accept it via their handler signature.
        """
