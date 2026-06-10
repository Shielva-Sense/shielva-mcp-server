"""
Shielva Assist — Tool Definitions & Executors

Defines the tools Claude can call to read data, create resources,
and control the CMS UI.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import os

import httpx
import structlog

from shielva_common.tls import internal_ca_verify
from src.routing.llm_router import ToolSpec

logger = structlog.get_logger(__name__)

CMS_CORE_URL = os.getenv("CMS_CORE_URL", "https://localhost:8020/api/v1/cms")


# ── Tool Handlers ─────────────────────────────────────────────────────

async def _cms_get(path: str, params: dict = None) -> Any:
    async with httpx.AsyncClient(verify=internal_ca_verify(), timeout=10.0) as c:
        r = await c.get(f"{CMS_CORE_URL}{path}", params=params or {})
        r.raise_for_status()
        return r.json()


async def _cms_post(path: str, body: dict) -> Any:
    async with httpx.AsyncClient(verify=internal_ca_verify(), timeout=10.0) as c:
        r = await c.post(f"{CMS_CORE_URL}{path}", json=body)
        r.raise_for_status()
        return r.json()


async def _cms_put(path: str, body: dict) -> Any:
    async with httpx.AsyncClient(verify=internal_ca_verify(), timeout=10.0) as c:
        r = await c.put(f"{CMS_CORE_URL}{path}", json=body)
        r.raise_for_status()
        return r.json()


# ── Read Tools ────────────────────────────────────────────────────────

async def handle_list_intents(tenant_id: str, **kw) -> dict:
    """List all intents for the tenant."""
    intents = await _cms_get("/intents", {"tenant_id": tenant_id})
    return {"intents": intents, "count": len(intents)}


async def handle_list_signal_rules(tenant_id: str, **kw) -> dict:
    """List all signal rules for the tenant."""
    rules = await _cms_get("/signals/rules", {"tenant_id": tenant_id})
    return {"rules": rules, "count": len(rules)}


async def handle_list_automations(tenant_id: str, **kw) -> dict:
    """List all automations for the tenant."""
    automations = await _cms_get("/automations", {"tenant_id": tenant_id})
    return {"automations": automations, "count": len(automations)}


async def handle_list_decision_rules(**kw) -> dict:
    """List all decision logic rules."""
    rules = await _cms_get("/logic")
    return {"rules": rules, "count": len(rules)}


async def handle_get_unmatched_signals(tenant_id: str, limit: int = 20, **kw) -> dict:
    """Get recent signal log entries that were unmatched."""
    log = await _cms_get("/signals/log", {"tenant_id": tenant_id, "limit": limit})
    unmatched = [e for e in log if e.get("routing_path") == "unmatched"]
    return {"unmatched_signals": unmatched, "count": len(unmatched)}


# ── Write Tools ───────────────────────────────────────────────────────

async def handle_create_intent(
    tenant_id: str,
    label: str,
    intent_id: str,
    examples: list = None,
    description: str = "",
    risk_level: str = "low",
    priority: int = 50,
    **kw,
) -> dict:
    """Create a new intent."""
    body = {
        "tenant_id": tenant_id,
        "intent_id": intent_id,
        "label": label,
        "description": description,
        "examples": examples or [],
        "required_entities": [],
        "priority": priority,
        "risk_level": risk_level,
        "automation_eligible": True,
        "enabled": True,
        "linked_rule_ids": [],
    }
    result = await _cms_post("/intents", body)
    return {"created": True, "intent": result}


async def handle_create_signal_rule(
    tenant_id: str,
    name: str,
    pattern: str,
    intent_id: str = "",
    rule_type: str = "keyword",
    priority: int = 50,
    **kw,
) -> dict:
    """Create a new signal rule."""
    body = {
        "tenant_id": tenant_id,
        "name": name,
        "rule_type": rule_type,
        "pattern": pattern,
        "intent_id": intent_id,
        "priority": priority,
        "enabled": True,
        "fuzzy_tolerance": 0.8,
    }
    result = await _cms_post("/signals/rules", body)
    return {"created": True, "rule": result}


async def handle_create_decision_rule(
    tenant_id: str,
    label: str,
    rule_id: str,
    action: str = "",
    condition: dict = None,
    priority: int = 50,
    **kw,
) -> dict:
    """Create a new decision logic rule."""
    body = {
        "tenant_id": tenant_id,
        "rule_id": rule_id,
        "label": label,
        "condition": condition or {},
        "action": action,
        "priority": priority,
        "enabled": True,
    }
    result = await _cms_post("/logic", body)
    return {"created": True, "rule": result}


async def handle_update_intent(
    intent_id: str,
    updates: dict,
    **kw,
) -> dict:
    """Update an existing intent by its MongoDB _id."""
    result = await _cms_put(f"/intents/{intent_id}", updates)
    return {"updated": True, "intent": result}


async def handle_toggle_intent(intent_id: str, enabled: bool, **kw) -> dict:
    """Enable or disable an intent."""
    result = await _cms_put(f"/intents/{intent_id}", {"enabled": enabled})
    return {"toggled": True, "intent": result}


# ── DOM Action Tools ──────────────────────────────────────────────────
# These don't execute server-side — they return action descriptors
# that the frontend executes as ClientActions.

async def handle_open_intent_form(
    prefill: dict = None,
    auto_save: bool = False,
    **kw,
) -> dict:
    """Open the intent creation form, optionally pre-filled."""
    return {
        "__client_action__": True,
        "action_type": "open_modal",
        "config": {
            "modal_id": "intent-create",
            "prefill": prefill or {},
            "auto_save": auto_save,
        },
    }


async def handle_navigate(url: str, **kw) -> dict:
    """Navigate to a CMS page."""
    return {
        "__client_action__": True,
        "action_type": "navigate",
        "config": {"url": url},
    }


async def handle_show_toast(message: str, variant: str = "success", **kw) -> dict:
    """Show a notification toast in the UI."""
    return {
        "__client_action__": True,
        "action_type": "show_toast",
        "config": {"message": message, "variant": variant},
    }


async def handle_highlight_element(selector: str, **kw) -> dict:
    """Highlight a UI element."""
    return {
        "__client_action__": True,
        "action_type": "highlight",
        "config": {"selector": selector, "duration_ms": 3000},
    }


# ── Tool Specs (for LLMRouter) ───────────────────────────────────────

ASSIST_TOOLS: List[ToolSpec] = [
    # Read tools
    ToolSpec(
        name="list_intents",
        description="List all intents configured for a tenant. Returns intent IDs, labels, examples count, risk levels.",
        parameters={
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string", "description": "The tenant ID"},
            },
            "required": ["tenant_id"],
        },
        handler=handle_list_intents,
    ),
    ToolSpec(
        name="list_signal_rules",
        description="List all signal rules for a tenant. Signal rules map text patterns to intents.",
        parameters={
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string", "description": "The tenant ID"},
            },
            "required": ["tenant_id"],
        },
        handler=handle_list_signal_rules,
    ),
    ToolSpec(
        name="list_automations",
        description="List all automations for a tenant. Automations are workflows triggered by intent matches.",
        parameters={
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string", "description": "The tenant ID"},
            },
            "required": ["tenant_id"],
        },
        handler=handle_list_automations,
    ),
    ToolSpec(
        name="list_decision_rules",
        description="List all decision logic rules. Decision rules define conditions and actions for matched intents.",
        parameters={
            "type": "object",
            "properties": {},
        },
        handler=handle_list_decision_rules,
    ),
    ToolSpec(
        name="get_unmatched_signals",
        description="Get recent signals (user messages) that didn't match any intent. Useful for discovering gaps.",
        parameters={
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string", "description": "The tenant ID"},
                "limit": {"type": "integer", "description": "Max results", "default": 20},
            },
            "required": ["tenant_id"],
        },
        handler=handle_get_unmatched_signals,
    ),

    # Write tools
    ToolSpec(
        name="create_intent",
        description="Create a new intent. An intent represents something a user is trying to do (e.g. 'check_balance'). Always provide examples.",
        parameters={
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string", "description": "The tenant ID"},
                "label": {"type": "string", "description": "Human-readable label (e.g. 'Check Balance')"},
                "intent_id": {"type": "string", "description": "Unique slug in snake_case (e.g. 'check_balance')"},
                "examples": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "3-10 example phrases users might say",
                },
                "description": {"type": "string", "description": "What this intent represents"},
                "risk_level": {"type": "string", "enum": ["low", "medium", "high", "critical"], "default": "low"},
                "priority": {"type": "integer", "description": "0-100, higher = matched first", "default": 50},
            },
            "required": ["tenant_id", "label", "intent_id", "examples"],
        },
        handler=handle_create_intent,
    ),
    ToolSpec(
        name="create_signal_rule",
        description="Create a signal rule that maps a text pattern to an intent.",
        parameters={
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "name": {"type": "string", "description": "Rule name"},
                "pattern": {"type": "string", "description": "The pattern to match"},
                "intent_id": {"type": "string", "description": "Intent to map to"},
                "rule_type": {"type": "string", "enum": ["exact", "regex", "keyword", "ngram"], "default": "keyword"},
                "priority": {"type": "integer", "default": 50},
            },
            "required": ["tenant_id", "name", "pattern"],
        },
        handler=handle_create_signal_rule,
    ),
    ToolSpec(
        name="create_decision_rule",
        description="Create a decision logic rule with conditions and actions.",
        parameters={
            "type": "object",
            "properties": {
                "tenant_id": {"type": "string"},
                "label": {"type": "string", "description": "Human-readable label"},
                "rule_id": {"type": "string", "description": "Unique slug in snake_case"},
                "action": {"type": "string", "description": "Action to take when condition matches"},
                "condition": {"type": "object", "description": "JSON condition logic"},
                "priority": {"type": "integer", "default": 50},
            },
            "required": ["tenant_id", "label", "rule_id"],
        },
        handler=handle_create_decision_rule,
    ),

    # DOM tools
    ToolSpec(
        name="open_intent_form",
        description="Open the intent creation form in the UI. Can optionally pre-fill fields and auto-save.",
        parameters={
            "type": "object",
            "properties": {
                "prefill": {
                    "type": "object",
                    "description": "Fields to pre-fill: intent_id, label, examples, risk_level, priority, description",
                },
                "auto_save": {"type": "boolean", "description": "Auto-submit the form after filling", "default": False},
            },
        },
        handler=handle_open_intent_form,
    ),
    ToolSpec(
        name="navigate",
        description="Navigate to a CMS page. URLs: /intents, /automation, /signals, /logic, /identity, /knowledge, /safety, /media, /",
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Relative URL path"},
            },
            "required": ["url"],
        },
        handler=handle_navigate,
    ),
    ToolSpec(
        name="show_toast",
        description="Show a notification toast message in the UI.",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "variant": {"type": "string", "enum": ["success", "error", "info"], "default": "success"},
            },
            "required": ["message"],
        },
        handler=handle_show_toast,
    ),
]
