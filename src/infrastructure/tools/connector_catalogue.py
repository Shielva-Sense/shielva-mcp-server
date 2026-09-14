"""A tenant's installed connectors, exposed as MCP tools.

🚨 THE TOOL SET IS PER TENANT, and that is the whole reason this is an adapter
rather than more registrations at startup. The built-in tools are identical for
everybody, so registering them once into a process-global registry is correct.
Connector tools are not: this workspace has Gmail and HubSpot installed, the
next has neither, and a tool list assembled at boot would offer every tenant all
216 connector types — including ones they have no credentials for, which is a
disclosure of nothing useful and an invitation to call something that will fail.

So the catalogue asks, per request, what THIS tenant has installed.

🚨 `list_for` IS NOT THE SECURITY BOUNDARY. `execute` is.

A client can call a tool it was never shown — `tools/list` is a discovery
convenience, and nothing in the protocol obliges a caller to have read it. If
the only check were in the listing, a client that guessed
`hubspot_connector__create_contact` would reach a connector its workspace never
installed, using whatever credential the runtime resolved. So the installed set
is checked AGAIN inside `execute`, against the same tenant, on every call. The
duplication is deliberate and load-bearing.

Two upstream calls make a listing:

  * ``GET /connectors``        — tenant-scoped, what is INSTALLED (7 rows here)
  * ``GET /connectors/types``  — deployment-wide, the SCHEMAS (216 rows)

The first is the authorisation fact and is never cached across tenants. The
second is a property of the deployment, not of a workspace, so it is cached
briefly — a paced run listing tools fifty times should not fetch 216 schemas
fifty times, and a newly published connector should still appear without a
restart.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from src.domain.shared.tenant import TenantContext
from src.domain.tools.entities import Tool
from src.domain.tools.repositories import ToolCatalogue, ToolExecutor
from src.domain.tools.value_objects import ToolName, ToolResult, ToolSchema

logger = structlog.get_logger(__name__)

#: Separates the connector from the method in a tool name. Double underscore
#: because MCP tool names are `[a-zA-Z0-9_-]` and connector types already carry
#: single underscores (`google_gmail_connector`) — a single separator could not
#: be split back apart unambiguously.
SEP = "__"

#: Methods that are plumbing, not capability. Exposing `install` or `authorize`
#: as callable tools would let a model try to re-run an OAuth handshake mid
#: conversation; `health_check` is noise in a tool list.
_NOT_TOOLS = frozenset({"install", "authorize", "health_check", "oauth2"})

#: The schema catalogue is a deployment fact, not a tenant one. Short enough
#: that a connector published minutes ago is usable without a restart.
_TYPES_TTL_S = 300.0

#: connector.json param types → JSON Schema types. Anything unrecognised is a
#: string: the connector will coerce or reject it, and guessing `object` for an
#: unknown type produces a schema the model cannot satisfy.
_JSON_TYPES = {
    "text": "string",
    "string": "string",
    "password": "string",
    "email": "string",
    "url": "string",
    "textarea": "string",
    "number": "number",
    "integer": "integer",
    "boolean": "boolean",
    "json": "object",
    "object": "object",
    "array": "array",
    "list": "array",
}


def tool_name_for(connector_type: str, api_id: str) -> ToolName:
    """`google_gmail_connector` + `send_email` → `google_gmail__send_email`.

    The `_connector` suffix is dropped because it is on every one of them and
    carries no information — it only spends characters in a name the model has
    to read and reproduce exactly.
    """
    base = connector_type[: -len("_connector")] if connector_type.endswith("_connector") else connector_type
    return ToolName(f"{base}{SEP}{api_id}")


def _split(name: str) -> tuple[str, str]:
    """A tool name back to (connector_type, method). ("", "") when it is not one."""
    base, sep, api_id = name.partition(SEP)
    if not sep or not base or not api_id:
        return "", ""
    return f"{base}_connector", api_id


def _schema_for(api: dict[str, Any]) -> ToolSchema:
    """One `apis[]` entry → the JSON Schema an LLM is handed."""
    props: dict[str, Any] = {}
    required: list[str] = []
    for p in api.get("params") or []:
        if not isinstance(p, dict) or not p.get("key"):
            continue
        key = str(p["key"])
        prop: dict[str, Any] = {"type": _JSON_TYPES.get(str(p.get("type") or "").lower(), "string")}
        # The label is the only human description most params carry, and a
        # parameter with no description is one the model fills by guessing.
        desc = str(p.get("help") or p.get("label") or "").strip()
        if desc:
            prop["description"] = desc
        props[key] = prop
        if p.get("required"):
            required.append(key)
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return ToolSchema(json_schema=schema)


class ConnectorToolCatalogue(ToolCatalogue, ToolExecutor):
    """The tenant's installed connectors, as MCP tools."""

    def __init__(self, *, base_url: str, timeout_s: float = 45.0, verify: Any = True) -> None:
        self._base = (base_url or "").rstrip("/")
        self._timeout = timeout_s
        self._verify = verify
        self._types_cache: tuple[float, dict[str, dict[str, Any]]] = (0.0, {})

    # ── upstream reads ────────────────────────────────────────────

    async def _types(self) -> dict[str, dict[str, Any]]:
        """`connector_type` → its declaration. Cached; see `_TYPES_TTL_S`."""
        now = time.monotonic()
        stamped, cached = self._types_cache
        if cached and now - stamped < _TYPES_TTL_S:
            return cached
        try:
            async with httpx.AsyncClient(verify=self._verify, timeout=self._timeout) as client:
                r = await client.get(f"{self._base}/connectors/types")
            if r.status_code >= 400:
                logger.warning("mcp.connector_types_unavailable", status=r.status_code)
                return cached
            body = r.json() if r.content else {}
            rows = body if isinstance(body, list) else (body.get("connector_types") or [])
        except Exception as exc:
            # The previous catalogue beats none: a blip must not empty every
            # tenant's tool list mid-conversation.
            logger.warning("mcp.connector_types_failed", error=str(exc)[:200])
            return cached
        types = {str(t.get("type")): t for t in rows if isinstance(t, dict) and t.get("type")}
        if types:
            self._types_cache = (now, types)
        return types

    async def _installed(self, tenant: TenantContext) -> set[str]:
        """Connector TYPES this tenant has installed.

        🚨 Never cached across tenants, and never cached at all: this is the
        authorisation fact. A connector uninstalled a moment ago must stop being
        callable now, not when a TTL expires.
        """
        try:
            async with httpx.AsyncClient(verify=self._verify, timeout=self._timeout) as client:
                r = await client.get(
                    f"{self._base}/connectors",
                    headers={"X-Tenant-ID": tenant.tenant_id},
                )
            if r.status_code >= 400:
                logger.warning(
                    "mcp.installed_connectors_unavailable",
                    tenant_id=tenant.tenant_id,
                    status=r.status_code,
                )
                return set()
            body = r.json() if r.content else []
            rows = body if isinstance(body, list) else (body.get("connectors") or [])
        except Exception as exc:
            # 🚨 Fails CLOSED, unlike the schema cache above. An empty tool list
            # is a feature that looks unavailable; a populated one we could not
            # authorise is a tenant calling a connector it may not own.
            logger.warning(
                "mcp.installed_connectors_failed",
                tenant_id=tenant.tenant_id,
                error=str(exc)[:200],
            )
            return set()
        return {str(row.get("connector_type")) for row in rows if isinstance(row, dict) and row.get("connector_type")}

    # ── ToolCatalogue ─────────────────────────────────────────────

    async def list_for(self, tenant: TenantContext) -> list[Tool]:
        installed = await self._installed(tenant)
        if not installed:
            return []
        types = await self._types()
        out: list[Tool] = []
        for ctype in sorted(installed):
            declared = types.get(ctype)
            if not declared:
                # Installed but no longer published — the catalogue snapshot
                # moved on. Listing it would offer a tool nothing can run.
                logger.info("mcp.installed_connector_not_in_catalogue", connector_type=ctype)
                continue
            for api in declared.get("apis") or []:
                api_id = str(api.get("id") or "")
                if not api_id or api_id in _NOT_TOOLS:
                    continue
                out.append(
                    Tool(
                        name=tool_name_for(ctype, api_id),
                        description=(
                            str(api.get("description") or api.get("name") or api_id).strip()
                            + f"  [{declared.get('display_name') or ctype}]"
                        ),
                        input_schema=_schema_for(api),
                        # The connector being installed IS the grant. A second
                        # permission string here would be a second place to
                        # keep in step with what ARC actually granted.
                        required_permissions=(),
                    )
                )
        return out

    async def get(self, name: ToolName) -> Tool | None:
        """Pure lookup, no tenant — the port says permission is the executor's
        job, and `execute` does exactly that."""
        ctype, api_id = _split(str(name))
        if not ctype:
            return None
        declared = (await self._types()).get(ctype)
        if not declared:
            return None
        for api in declared.get("apis") or []:
            if str(api.get("id") or "") == api_id and api_id not in _NOT_TOOLS:
                return Tool(
                    name=name,
                    description=str(api.get("description") or api_id),
                    input_schema=_schema_for(api),
                )
        return None

    # ── ToolExecutor ──────────────────────────────────────────────

    async def execute(
        self,
        *,
        tool: Tool,
        arguments: dict[str, Any],
        tenant: TenantContext,
        context: dict[str, Any] | None = None,
    ) -> ToolResult:
        ctype, method = _split(str(tool.name))
        if not ctype:
            return ToolResult.text(f"{tool.name} is not a connector tool.", is_error=True)

        # 🚨 THE authorisation check. `list_for` filtered the listing, and a
        # listing is not a boundary — nothing obliges a caller to have read it,
        # so a guessed name would otherwise reach a connector this workspace
        # never installed.
        if ctype not in await self._installed(tenant):
            logger.warning(
                "mcp.connector_tool_not_installed",
                tenant_id=tenant.tenant_id,
                connector_type=ctype,
                tool=str(tool.name),
            )
            return ToolResult.text(
                f"The {ctype} connector is not installed for this workspace.",
                is_error=True,
            )

        try:
            async with httpx.AsyncClient(verify=self._verify, timeout=self._timeout) as client:
                r = await client.post(
                    f"{self._base}/connectors/{ctype}/test/{method}",
                    json=arguments or {},
                    headers={"X-Tenant-ID": tenant.tenant_id},
                )
        except Exception as exc:
            logger.warning(
                "mcp.connector_tool_unreachable",
                tenant_id=tenant.tenant_id,
                tool=str(tool.name),
                error=str(exc)[:200],
            )
            return ToolResult.text(f"{ctype} could not be reached.", is_error=True)

        body = r.json() if r.content else {}
        if r.status_code >= 400:
            detail = str(body.get("detail") or body.get("error") or f"HTTP {r.status_code}")[:300]
            return ToolResult.text(f"{ctype} refused: {detail}", is_error=True)

        # 🚨 The runtime answers 200 with `{"status": "error"}` when a connector
        # raised or timed out. A caller checking only the HTTP code reports
        # those as successes, and the model then tells somebody their email
        # was sent.
        if isinstance(body, dict) and str(body.get("status") or "").lower() == "error":
            detail = str(body.get("message") or body.get("error") or "the connector reported an error")[:300]
            return ToolResult.text(f"{ctype} reported: {detail}", is_error=True)

        import json as _json

        result = body.get("result", body) if isinstance(body, dict) else body
        return ToolResult.text(_json.dumps(result, default=str)[:20_000])
