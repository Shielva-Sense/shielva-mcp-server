"""Which tools a bot gets by DEFAULT.

Every tool used to ship ``enabled_by_default=True`` with no permission gate, and
the per-bot config that was meant to narrow it (``configure_bot_tools``) had no
callers. Net effect: every bot in every tenant was handed all of them — including
four that CREATE records in TMS, and ``get_delegation_rules``, whose description
tells the model to call it before answering anything. A knowledge bot would obey,
the call would fail, and it would refuse to answer.

Meanwhile ``rag_query`` — the only tool it actually needed — was the ONLY one with
a permission requirement (``rag_access``), resolved against a collection that is
empty in production. So the one useful tool was the one switched off.

These tests pin the corrected shape. They are about blast radius, not plumbing:
a customer-facing bot must not be able to write to TMS by default.
"""

from __future__ import annotations

import ast
import pathlib

_TOOL_SOURCES = [
    *sorted(pathlib.Path("src/tools").glob("*.py")),
    pathlib.Path("src/registry/tool_registry.py"),
]

# Tools that write to, or read privileged context from, another product surface.
# None of these belong on a general bot unless explicitly turned on for it.
_MUST_BE_OPT_IN = {
    "create_tms_goal",
    "create_tms_epic",
    "create_tms_sprint",
    "create_tms_ticket",
    "meeting_context_query",
    "get_delegation_rules",
    "codegen_validate_python",
    "codegen_analyze_imports",
    "codegen_categorize_error",
    "codegen_check_pytest_structure",
}


def _tool_defs() -> dict[str, dict]:
    """Read every ToolDefinition literal without importing the modules (they pull
    in infra); the declaration IS the contract under test."""
    out: dict[str, dict] = {}
    for f in _TOOL_SOURCES:
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ToolDefinition":
                kw = {k.arg: k.value for k in node.keywords}
                name = getattr(kw.get("name"), "value", None)
                if not name:
                    continue
                enabled = kw.get("enabled_by_default")
                perms = kw.get("requires_permissions")
                out[name] = {
                    # The field itself defaults to True, so an ABSENT flag means ON.
                    "enabled": True if enabled is None else getattr(enabled, "value", True),
                    "perms": [getattr(e, "value", "?") for e in perms.elts] if isinstance(perms, ast.List) else [],
                }
    return out


def test_privileged_tools_are_not_on_by_default():
    defs = _tool_defs()
    leaked = sorted(n for n in _MUST_BE_OPT_IN if defs.get(n, {}).get("enabled"))
    assert not leaked, f"these must be opt-in per bot, not default-on: {leaked}"


def test_tms_write_tools_are_never_default_on():
    """The sharpest edge: a public chat bot able to create Goals/Epics/Sprints/
    Tickets in the tenant's TMS."""
    defs = _tool_defs()
    for name in ("create_tms_goal", "create_tms_epic", "create_tms_sprint", "create_tms_ticket"):
        assert defs[name]["enabled"] is False, f"{name} would be handed to every bot"


def test_knowledge_search_is_available_by_default():
    """The regression that made bots useless: rag_query gated behind a permission
    that lives in an unpopulated collection, so it resolved to False for everyone."""
    defs = _tool_defs()
    assert defs["rag_query"]["enabled"] is True
    assert defs["rag_query"]["perms"] == [], (
        "rag_query must not require a permission the platform never grants — "
        "knowledge access is decided by the KBs linked to the bot"
    )


def test_default_tool_set_is_minimal():
    """A general bot should carry only what any bot needs. Anything else being
    added here should be a deliberate decision, which is what this test forces."""
    defs = _tool_defs()
    on = {n for n, d in defs.items() if d["enabled"]}
    assert on == {"rag_query", "get_current_time"}, f"unexpected default tool set: {sorted(on)}"


def test_no_tool_declares_a_permission_the_platform_cannot_grant():
    """`rag_access` was never in the identity permission catalog (bots/kbs/tms/acp/
    ...). A requirement nothing can satisfy is a silent deny, not a control."""
    defs = _tool_defs()
    for name, d in defs.items():
        assert "rag_access" not in d["perms"], f"{name} still requires the ungrantable rag_access"


def test_internal_tools_are_not_advertised_to_a_tenant():
    """🚨 A platform owner connecting Claude saw fifteen tools that are not
    product: codegen_* (the fix-agent endpoint) and create_tms_* (post-meeting
    transcript extraction, and TMS is not even in production any more).

    Every one of them declared `enabled_by_default=False` — the author saying
    "not on for everyone" — and NOTHING read it. `Tool.is_permitted_for`
    consults `required_permissions` only, and these declared none, which the
    entity treats as public to every authenticated tenant. A declaration that
    enforces nothing is worse than no declaration: it reads like a control.
    """
    from src.tools.codegen_tools import INTERNAL_ONLY as CODEGEN_ONLY
    from src.tools.meeting_tools import INTERNAL_ONLY as MEETING_ONLY

    assert CODEGEN_ONLY, "codegen tools must require something"
    assert MEETING_ONLY, "meeting tools must require something"

    from src.domain.shared.tenant import TenantContext as DomainTenant
    from src.domain.tools.entities import Tool
    from src.domain.tools.value_objects import ToolName, ToolSchema

    gated = Tool(
        name=ToolName("create_tms_epic"),
        description="x",
        input_schema=ToolSchema(json_schema={"type": "object"}),
        required_permissions=tuple(MEETING_ONLY),
    )
    customer = DomainTenant(
        tenant_id="t1", user_id="u1", user_email="someone@example.com", role="customer", permissions=()
    )
    assert not gated.is_permitted_for(customer), "a customer can still see an internal tool"


def test_every_codegen_and_tms_tool_declares_the_internal_permission():
    """Belt and braces: one forgotten entry re-publishes that tool to everyone."""
    import re
    from pathlib import Path

    for path in ("src/tools/codegen_tools.py", "src/tools/meeting_tools.py"):
        src = Path(path).read_text()
        assert "requires_permissions=[]" not in src, f"{path} still has an ungated tool"
        assert re.search(r"requires_permissions=INTERNAL_ONLY", src), path
