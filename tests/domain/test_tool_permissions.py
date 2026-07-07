"""Tool entity — permission filtering.

These tests assert the invariant that tools with empty
``required_permissions`` are public to every authenticated tenant,
and tools with a permission list require that EVERY listed
permission be present (AND semantics, not OR — adapters that want
OR semantics should compose two Tool entries or one with no perms
plus a server-side check).
"""

from __future__ import annotations

from src.domain.shared.tenant import TenantContext
from src.domain.tools.entities import Tool
from src.domain.tools.value_objects import ToolName, ToolSchema


def _tenant(*perms: str) -> TenantContext:
    return TenantContext(
        tenant_id="t1",
        user_id="u1",
        user_email="u@example.com",
        permissions=tuple(perms),
    )


def _tool(*required: str) -> Tool:
    return Tool(
        name=ToolName("test_tool"),
        description="x",
        input_schema=ToolSchema(json_schema={"type": "object"}),
        required_permissions=tuple(required),
    )


class TestToolPermissions:
    def test_no_required_perms_public(self) -> None:
        assert _tool().is_permitted_for(_tenant()) is True

    def test_single_perm_present_allowed(self) -> None:
        assert _tool("kb:read").is_permitted_for(_tenant("kb:read")) is True

    def test_single_perm_missing_denied(self) -> None:
        assert _tool("kb:write").is_permitted_for(_tenant("kb:read")) is False

    def test_multiple_perms_and_semantics(self) -> None:
        t = _tool("kb:read", "tools:invoke")
        assert t.is_permitted_for(_tenant("kb:read")) is False
        assert t.is_permitted_for(_tenant("kb:read", "tools:invoke")) is True
        assert t.is_permitted_for(_tenant("kb:read", "tools:invoke", "extra")) is True

    def test_empty_tenant_perms_blocks_restricted_tool(self) -> None:
        assert _tool("admin").is_permitted_for(_tenant()) is False
