"""Adapter wrapping the legacy ToolRegistry into the new domain ports.

Why this adapter (and not a fresh re-implementation)
----------------------------------------------------
The existing ``src.registry.tool_registry.ToolRegistry`` is shared
state across multiple consumers:

    * ``api/codegen_routes.py`` — uses ``app.state.tool_registry`` to
      build ToolSpec lists for the fix-agent loop.
    * ``protocol/message_handler.py`` — same registry feeds
      ``handle_query`` (slice 3 will move this).
    * Codegen + meeting tool registration at startup
      (``register_codegen_tools(registry)`` /
      ``register_meeting_tools(registry)``).

If we re-implemented tool storage in the new layer, those consumers
would still write to the *old* registry, and the new path would see
an empty catalogue. The cleanest interim move is to wrap the
existing registry behind the new ports — one source of truth, two
APIs over it. Slice 4 will lift the remaining legacy callers and
delete the wrapper.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import structlog

from src.domain.shared.tenant import TenantContext
from src.domain.tools.entities import Tool
from src.domain.tools.errors import (
    ToolNotFoundError, ToolPermissionDeniedError, ToolExecutionError,
)
from src.domain.tools.repositories import ToolCatalogue, ToolExecutor
from src.domain.tools.value_objects import (
    ToolName, ToolResult, ToolSchema, ToolText,
)

logger = structlog.get_logger(__name__)


class LegacyToolRegistryAdapter(ToolCatalogue, ToolExecutor):
    """Implements both ports against the existing in-memory registry.

    The legacy registry is constructed by ``main.py`` lifespan and
    populated by the codegen + meeting registration hooks. We hold a
    reference to that singleton — no second instantiation. The
    composition root gets the legacy registry off ``app.state``.
    """

    def __init__(self, legacy_registry: Any) -> None:
        self._reg = legacy_registry

    # ── ToolCatalogue ─────────────────────────────────────────────

    async def list_for(self, tenant: TenantContext) -> List[Tool]:
        """All tools the tenant is permitted to invoke.

        The legacy registry's ``_check_permissions`` is reused so
        permission semantics stay identical to the codegen path.
        """
        legacy_tenant = _to_legacy_tenant(tenant)
        out: List[Tool] = []
        for definition in self._reg.get_all_tools():
            if not self._reg._check_permissions(definition, legacy_tenant):  # noqa: SLF001
                continue
            out.append(self._to_domain(definition))
        return out

    async def get(self, name: ToolName) -> Optional[Tool]:
        registered = self._reg._tools.get(str(name))  # noqa: SLF001
        if registered is None:
            return None
        return self._to_domain(registered.definition)

    # ── ToolExecutor ──────────────────────────────────────────────

    async def execute(
        self,
        *,
        tool:      Tool,
        arguments: Dict[str, Any],
        tenant:    TenantContext,
        context:   Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        # Translate to the legacy execute_tool envelope. We import
        # the legacy types lazily to keep the dependency footprint
        # narrow — once slice 4 deletes the wrapper, these imports
        # vanish with it.
        from src.protocol.models import (
            ToolExecutionRequest, ToolExecutionResponse,
        )

        legacy_tenant = _to_legacy_tenant(tenant)
        # Defence-in-depth — the application service already
        # permission-checked, but the legacy registry also runs its
        # own check inside execute_tool. We catch its denial and
        # re-raise as the domain error so the use-case layer sees
        # the same exception shape regardless of adapter.
        #
        # Shielva-internal service callers bypass tenant tool-permission gating
        # (matches Tool.is_permitted_for + the un-gated /codegen/* endpoints):
        # internal infra like the integration builder's codegen-guideline RAG
        # is not tenant feature-use. Identity is gateway-controlled (not forgeable
        # by a tenant), so this does not widen the tenant-facing surface.
        if not tenant.is_internal_service and not self._reg._check_permissions(  # noqa: SLF001
            self._reg._tools[str(tool.name)].definition,  # noqa: SLF001
            legacy_tenant,
        ):
            raise ToolPermissionDeniedError(
                f"Tenant lacks permission for tool {tool.name}",
            )

        req = ToolExecutionRequest(
            tool_name  = str(tool.name),
            parameters = dict(arguments or {}),
            context    = dict(context or {}),
        )
        try:
            resp: ToolExecutionResponse = await self._reg.execute_tool(req, legacy_tenant)
        except Exception as exc:  # noqa: BLE001
            # Legacy registry SHOULD catch internal errors, but
            # belt-and-braces.
            raise ToolExecutionError(str(exc)) from exc

        if not resp.success:
            # Per spec: errored tools surface to the LLM via the
            # content array, not as JSON-RPC errors. The use-case
            # layer reads ``is_error`` and acts accordingly.
            return ToolResult.failure(resp.error or "tool failed")

        payload = resp.result
        if isinstance(payload, str):
            text = payload
        else:
            try:
                text = json.dumps(payload, ensure_ascii=False, default=str)
            except Exception:
                text = str(payload)
        return ToolResult.text(text)

    # ── internal ──────────────────────────────────────────────────

    def _to_domain(self, definition: Any) -> Tool:
        """Convert legacy ``ToolDefinition`` (Pydantic) to the new
        :class:`Tool` entity. The legacy registry's private
        ``_build_parameters_schema`` produces the JSON Schema dict
        the LLM expects — we reuse it so schema semantics don't
        drift."""
        schema = self._reg._build_parameters_schema(definition)  # noqa: SLF001
        return Tool(
            name                 = ToolName(definition.name),
            description          = definition.description,
            input_schema         = ToolSchema(json_schema=schema),
            required_permissions = tuple(definition.requires_permissions or ()),
            enabled_by_default   = bool(definition.enabled_by_default),
        )


def _to_legacy_tenant(tenant: TenantContext):
    """Build the legacy Pydantic TenantContext that ``execute_tool``
    and ``_check_permissions`` consume. Once the legacy registry is
    deleted in slice 4 this helper goes with it."""
    from src.protocol.models import TenantContext as Legacy
    return Legacy(
        tenant_id   = tenant.tenant_id,
        user_id     = tenant.user_id,
        user_email  = tenant.user_email,
        role        = tenant.role,
        permissions = list(tenant.permissions),
    )
