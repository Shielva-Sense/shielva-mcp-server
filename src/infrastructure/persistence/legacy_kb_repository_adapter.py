"""Adapter wrapping the existing in-memory KBRegistry behind the
:class:`domain.knowledge.KBRepository` port.

The legacy registry only exposes ``register`` + ``get_kbs_for_bot``
and its in-memory state is sparse — most installations expect Mongo-
backed metadata, which slice 4 will deliver as a sibling Mongo
adapter. For now this wrapper:
  * Returns ``None`` from ``get`` when the legacy registry doesn't
    track per-id lookup (it currently doesn't).
  * Surfaces ``list_for_tenant`` as an aggregate over the registry's
    internal store, falling back to empty when the registry has no
    such accessor.

This is intentionally minimal — the slice's goal is the *port*, not
a full KB metadata layer. The MCP JSON-RPC ``resources/list`` path
calls this adapter; with an empty registry it returns ``[]`` (which
is exactly what the spec smoke validates).
"""
from __future__ import annotations

from typing import List, Optional

import structlog

from src.domain.knowledge.entities import KnowledgeBase
from src.domain.knowledge.repositories import KBRepository
from src.domain.knowledge.value_objects import KBId, KBName, KBStatus
from src.domain.shared.tenant import TenantContext

logger = structlog.get_logger(__name__)


class LegacyKBRepositoryAdapter(KBRepository):
    def __init__(self, legacy_registry) -> None:
        self._reg = legacy_registry

    async def get(self, kb_id: KBId, *, tenant: TenantContext
                  ) -> Optional[KnowledgeBase]:
        for kb in await self.list_for_tenant(tenant=tenant):
            if kb.id == kb_id:
                return kb
        return None

    async def list_for_tenant(self, *, tenant: TenantContext
                              ) -> List[KnowledgeBase]:
        raw = await _scan_legacy_store(self._reg, tenant.tenant_id)
        return [_to_domain(r, tenant.tenant_id) for r in raw]

    async def list_for_bot(self, *, bot_id: str, tenant: TenantContext
                           ) -> List[KnowledgeBase]:
        # Legacy registry's per-bot accessor returns dicts in the
        # same shape; just route through it.
        try:
            raw = await self._reg.get_kbs_for_bot(bot_id, tenant.tenant_id)
        except Exception as e:  # noqa: BLE001
            logger.warning("kb_registry.get_kbs_for_bot failed", error=str(e))
            raw = []
        return [_to_domain(r, tenant.tenant_id, bot_id=bot_id) for r in (raw or [])]

    async def save(self, kb: KnowledgeBase) -> None:
        # ProvisionKB use case (slice 4) will hit this — for now the
        # legacy registry doesn't expose an upsert beyond
        # ``register`` which takes a Pydantic KBConfig. We log and
        # skip; tests that exercise save() use a fake repo anyway.
        logger.info(
            "mcp.kb_save_unsupported",
            kb_id=str(kb.id),
            note="Legacy KBRegistry has no upsert path; provision flow uses register()",
        )


# ── helpers ────────────────────────────────────────────────────────

async def _scan_legacy_store(registry, tenant_id: str) -> list:
    """Best-effort tenant-wide scan. Mirrors the legacy bridges in
    the slice-1 dispatcher so behaviour stays consistent during the
    migration."""
    listf = getattr(registry, "list_for_tenant", None)
    if listf is not None:
        try:
            return list(await listf(tenant_id))
        except Exception:
            pass
    store = getattr(registry, "_kbs", None)
    if isinstance(store, dict):
        out: list = []
        for entry in store.values():
            if isinstance(entry, list):
                out.extend([k for k in entry if (k.get("tenant_id") or "") == tenant_id])
            elif isinstance(entry, dict) and (entry.get("tenant_id") or "") == tenant_id:
                out.append(entry)
        return out
    return []


def _to_domain(raw: dict, tenant_id: str, *, bot_id: str = "") -> KnowledgeBase:
    return KnowledgeBase(
        id          = KBId(str(raw.get("kb_id") or raw.get("id") or "")),
        tenant_id   = tenant_id,
        bot_id      = bot_id or str(raw.get("bot_id") or ""),
        name        = KBName(str(raw.get("name") or raw.get("unique_name") or "")),
        status      = _parse_status(raw.get("status")),
        hello_id    = str(raw.get("hello_id") or ""),
        unique_name = str(raw.get("unique_name") or ""),
        doc_count   = int(raw.get("doc_count") or raw.get("documents") or 0),
        metadata    = dict(raw.get("metadata") or {}),
    )


def _parse_status(value) -> KBStatus:
    if value is None:
        return KBStatus.PENDING
    try:
        return KBStatus(str(value))
    except ValueError:
        return KBStatus.PENDING
