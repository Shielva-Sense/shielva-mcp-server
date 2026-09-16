"""Every configuration tool sends a body its backend model actually accepts.

🚨 WHY THIS EXISTS: twelve of the MCP configuration tools were written against
imagined request shapes, and it was invisible until someone tried to build a bot.

    create_identity       sent persona, never role        -> 422 "role: Field required"
    create_intent         sent name + bot_id              -> no intent_id / label
    create_decision_rule  spread an unchecked dict        -> 422
    create_action_schema  sent name/connector_type/...    -> none of action_id/label/source_type
    test_action_schema    sent schema_id + inputs         -> no action_id
    set_bot_prompt        sent prompt                     -> 400 "system_prompt is required"
    set_bot_signals       sent signals                    -> SILENTLY stored [] (default)
    update_*              PUT partial changes             -> 422 on full-replace models

Pydantic ignores unknown keys, so a wrong field name does not error — it is just
dropped. Only the REQUIRED fields fail loudly, and set_bot_signals did not even do
that. So each tool is checked both ways: every required field present, and no key
the model does not define.

The field sets below were read from the live services (acp-core
app/schemas/*.py and core-api app/api/bots.py). If a backend model changes, update
them here — and the tool with them.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.protocol.models import TenantContext
from src.tools import bot_config_tools as cfg

# ── the real backend models ──────────────────────────────────────────────────

IDENTITY_CREATE = {
    "tenant_id", "bot_id", "name", "role", "personality_traits", "tone", "boundaries",
    "greeting", "fallback_message", "version", "is_active", "page_scopes",
}  # fmt: skip
IDENTITY_PATCH = {"bot_id", "name", "role", "personality_traits", "tone", "boundaries", "greeting", "fallback_message"}
INTENT_CREATE = {
    "tenant_id", "intent_id", "label", "description", "examples", "required_entities", "priority",
    "risk_level", "automation_eligible", "enabled", "linked_rule_ids", "linked_action_id",
    "linked_painter_id", "painter_data_mode", "painter_field_map", "ai_generated",
}  # fmt: skip
SIGNAL_RULE = {
    "tenant_id", "name", "description", "rule_type", "pattern", "intent_id", "priority",
    "enabled", "fuzzy_tolerance", "ai_generated",
}  # fmt: skip
ACTION_SCHEMA_CREATE = {
    "tenant_id", "action_id", "label", "description", "source_type", "confirm_message",
    "content_config", "form_config", "asset_config", "api_config", "webhook_binding",
    "response_mapping", "painter_id", "painter_layout", "field_configs", "layout_style",
    "table_filters", "enabled", "version", "test_environment", "custom_design",
}  # fmt: skip
API_CONFIG = {
    "connector_id", "endpoint", "method", "response_type", "poll_interval_ms", "async_status_endpoint",
    "ws_message_path", "ws_endpoint", "ws_auth_type", "ws_reconnect", "ws_reconnect_delay_ms",
    "sse_event_type", "param_mapping", "headers",
}  # fmt: skip
ACTION_TEST_REQUEST = {"tenant_id", "action_id", "sample_entities", "sample_context", "sample_payload"}
PAINTER_CREATE = {"tenant_id", "name", "painter_type", "config", "description", "status"}


def _t() -> TenantContext:
    return TenantContext(tenant_id="Tenant-x", user_id="u1", user_email="u@example.com")


class _Gateway:
    """Records each request; answers a GET with a stored record for merge-PUTs."""

    def __init__(self, record: dict[str, Any] | None = None) -> None:
        self.record = record or {}
        self.calls: list[tuple[str, str, Any]] = []

    async def __call__(self, method, path, tenant_context, *, params=None, json=None):
        self.calls.append((method, path, json))
        if method == "GET":
            return dict(self.record)
        return {"_id": "doc-1", **(json or {})} if isinstance(json, dict) else {}

    def last_write(self) -> tuple[str, str, dict[str, Any]]:
        return next(c for c in reversed(self.calls) if c[0] != "GET")  # type: ignore[return-value]


def _check(body: dict[str, Any], allowed: set[str], required: set[str]) -> None:
    missing = required - set(body)
    unknown = set(body) - allowed
    assert not missing, f"required fields not sent: {sorted(missing)}"
    assert not unknown, f"fields the backend model does not define (silently dropped): {sorted(unknown)}"


def _run(monkeypatch, coro_factory, record=None):
    gw = _Gateway(record)
    monkeypatch.setattr(cfg, "_call", gw)
    out = asyncio.run(coro_factory())
    return gw, out


# ── identity ──────────────────────────────────────────────────────────────────


def test_create_identity_sends_role_and_no_persona_field(monkeypatch):
    gw, out = _run(
        monkeypatch,
        lambda: cfg.shielva_create_identity(
            _t(), "Assistant", "b1", persona="Lettings assistant for a London agency. Books viewings.", tone="warm"
        ),
    )
    method, path, body = gw.last_write()
    assert (method, path) == ("POST", "/cms/api/v1/identity")
    _check(body, IDENTITY_CREATE, {"name", "role"})
    assert body["role"] == "Lettings assistant for a London agency"
    assert out["status"] == "created"
    assert out["identity_id"] == "doc-1"


def test_explicit_role_wins_over_persona(monkeypatch):
    gw, _ = _run(
        monkeypatch, lambda: cfg.shielva_create_identity(_t(), "A", "b1", persona="long text", role="Concierge")
    )
    assert gw.last_write()[2]["role"] == "Concierge"


def test_update_identity_patches_instead_of_replacing(monkeypatch):
    gw, _ = _run(
        monkeypatch, lambda: cfg.shielva_update_identity(_t(), "id1", {"tone": "calm", "persona": "Receptionist"})
    )
    method, path, body = gw.last_write()
    assert (method, path) == ("PATCH", "/cms/api/v1/identity/id1")
    _check(body, IDENTITY_PATCH, set())
    assert body["role"] == "Receptionist"


# ── intents ───────────────────────────────────────────────────────────────────


def test_create_intent_sends_intent_id_and_label(monkeypatch):
    gw, out = _run(
        monkeypatch, lambda: cfg.shielva_create_intent(_t(), "Book a Viewing", "b1", examples=["can I see it"])
    )
    method, path, body = gw.last_write()
    assert (method, path) == ("POST", "/cms/api/v1/intents")
    _check(body, INTENT_CREATE, {"intent_id", "label"})
    assert body["intent_id"] == "book_a_viewing"
    assert out["intent_id"] == "book_a_viewing", "a classify case needs this exact slug"


def test_update_intent_merges_onto_the_full_record(monkeypatch):
    record = {"_id": "d1", "intent_id": "x", "label": "X", "examples": ["a"], "tenant_id": "T", "created_at": "t"}
    gw, _ = _run(monkeypatch, lambda: cfg.shielva_update_intent(_t(), "d1", {"examples": ["a", "b"]}), record)
    method, path, body = gw.last_write()
    assert (method, path) == ("PUT", "/cms/api/v1/intents/d1")
    _check(body, INTENT_CREATE, {"intent_id", "label"})
    assert body["examples"] == ["a", "b"]


# ── decision rules ────────────────────────────────────────────────────────────


def test_create_decision_rule_requires_its_fields_before_calling(monkeypatch):
    gw, out = _run(monkeypatch, lambda: cfg.shielva_create_decision_rule(_t(), "r", {"pattern": "price"}))
    assert out["status"] == "failed"
    assert gw.calls == []
    assert "rule_type" in out["error"]
    assert "intent_id" in out["error"]


def test_create_decision_rule_sends_a_valid_signal_rule(monkeypatch):
    gw, _ = _run(
        monkeypatch,
        lambda: cfg.shielva_create_decision_rule(
            _t(), "price", {"rule_type": "keyword", "pattern": "price,cost", "intent_id": "ask_price"}
        ),
    )
    _check(gw.last_write()[2], SIGNAL_RULE, {"name", "rule_type", "pattern", "intent_id"})


def test_update_decision_rule_merges(monkeypatch):
    record = {"_id": "r1", "name": "n", "rule_type": "exact", "pattern": "p", "intent_id": "i"}
    gw, _ = _run(monkeypatch, lambda: cfg.shielva_update_decision_rule(_t(), "r1", {"priority": 5}), record)
    _check(gw.last_write()[2], SIGNAL_RULE, {"name", "rule_type", "pattern", "intent_id"})


# ── bot prompt + signals ──────────────────────────────────────────────────────


def test_set_bot_prompt_sends_system_prompt(monkeypatch):
    gw, _ = _run(monkeypatch, lambda: cfg.shielva_set_bot_prompt(_t(), "b1", "You are..."))
    assert gw.last_write()[2] == {"system_prompt": "You are..."}


def test_set_bot_signals_sends_rule_ids_and_never_silently_clears(monkeypatch):
    gw, _ = _run(monkeypatch, lambda: cfg.shielva_set_bot_signals(_t(), "b1", ["r1", {"_id": "r2"}]))
    assert gw.last_write()[2] == {"enabled_signal_rule_ids": ["r1", "r2"]}

    gw, out = _run(monkeypatch, lambda: cfg.shielva_set_bot_signals(_t(), "b1", [{"name": "no id"}]))
    assert out["status"] == "failed"
    assert gw.calls == [], "a list with no ids must not become an empty write"


# ── action schemas ────────────────────────────────────────────────────────────


def test_create_action_schema_is_an_api_schema_with_a_valid_api_config(monkeypatch):
    gw, out = _run(
        monkeypatch,
        lambda: cfg.shielva_create_action_schema(
            _t(), "Find Listings", "gravity_listings", "search", inputs={"area": "{{entity.area}}"}
        ),
    )
    method, path, body = gw.last_write()
    assert (method, path) == ("POST", "/cms/api/v1/action-schemas")
    _check(body, ACTION_SCHEMA_CREATE, {"action_id", "label", "source_type"})
    assert body["source_type"] == "api"
    _check(body["api_config"], API_CONFIG, {"connector_id", "endpoint"})
    assert out["action_id"] == "find_listings"


def test_update_action_schema_merges(monkeypatch):
    record = {"_id": "s1", "action_id": "a", "label": "A", "source_type": "api"}
    gw, _ = _run(monkeypatch, lambda: cfg.shielva_update_action_schema(_t(), "s1", {"description": "d"}), record)
    _check(gw.last_write()[2], ACTION_SCHEMA_CREATE, {"action_id", "label", "source_type"})


def test_test_action_schema_sends_action_id(monkeypatch):
    gw, _ = _run(monkeypatch, lambda: cfg.shielva_test_action_schema(_t(), "find_listings", {"area": "Marina"}))
    _check(gw.last_write()[2], ACTION_TEST_REQUEST, {"action_id"})


# ── painters ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("custom", [False, True])
def test_create_painter_puts_template_in_config(monkeypatch, custom):
    gw, _ = _run(
        monkeypatch, lambda: cfg.shielva_create_painter(_t(), "Listing card", "card", template={"x": 1}, custom=custom)
    )
    body = gw.last_write()[2]
    _check(body, PAINTER_CREATE, {"name", "painter_type"})
    assert body["config"] == {"x": 1}
    assert body["painter_type"] == ("custom" if custom else "card")
