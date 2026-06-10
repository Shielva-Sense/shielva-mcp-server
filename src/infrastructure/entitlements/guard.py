"""Subscription entitlement gate — the LLM broker's single enforcement point.

Every LLM-serving MCP endpoint asks shielva-billing "is tenant X entitled to
feature Y?" before serving. billing owns the managed-vs-self_hosted decision
(a provisioned subscription fans plan entitlements → active; an unprovisioned
self-hosted tenant has none → not entitled), so MCP needs **no** deployment_type
of its own — it just enforces billing's answer.

Billing contract:
    GET {billing_url}/v1/subscriptions/{tenant_id}/entitlements/{feature_key}
        → 200 {"entitled": bool, ...}   (200 even when there is no subscription)

Design choices:
  * **Disabled by default** (``entitlement_enabled``). A new gate on the hot
    path is opt-in per deployment, enabled once billing is populated — so it can
    never silently 403 all LLM traffic the moment it ships.
  * **Bypass** for Shielva-internal admin principals (super_admin /
    platform_owner / shielva_admin). Service-to-service calls carry their service
    role (e.g. cms_service), so they correctly fall through to the tenant check.
  * **Short-TTL in-process cache** (per pod) so billing is not called on every
    token. Staleness window = ``entitlement_cache_ttl_seconds``.
  * **Fail mode**: billing unreachable → FAIL OPEN by default (a billing outage
    must not take down all LLM traffic) + warn + metric; flip with
    ``entitlement_fail_closed`` for strict deployments.
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import time
from typing import Dict, Optional, Tuple

import httpx
import structlog
from fastapi import HTTPException, Request

from shielva_common.auth import Principal, require_principal

from config.settings import get_settings

logger = structlog.get_logger(__name__)

# Shielva-internal principals that bypass the tenant entitlement check.
_BYPASS_ROLES = ("super_admin", "platform_owner", "shielva_admin")

# MCP's own service identity for the billing S2S call (NOT a tenant, NOT a key).
_SVC_USER_ID = "mcp-svc"
_SVC_EMAIL = "mcp@shielva.internal"
_SVC_ROLES = "mcp_service"
_SVC_AUTH_METHOD = "service"
_SVC_IDENTITY = "shielva-mcp"


def _service_headers(tenant_id: str) -> Dict[str, str]:
    """S2S principal headers for MCP → billing, signed when the HMAC key is
    configured (works in billing's ``strict`` principal mode; unsigned is fine
    in ``verify`` mode)."""
    tid = tenant_id or ""
    headers: Dict[str, str] = {
        "X-Shielva-User-Id": _SVC_USER_ID,
        "X-Shielva-Email": _SVC_EMAIL,
        "X-Shielva-Tenant-Id": tid,
        "X-Shielva-Roles": _SVC_ROLES,
        "X-Shielva-Auth-Method": _SVC_AUTH_METHOD,
        "X-User-Id": _SVC_USER_ID,
        "X-User-Email": _SVC_EMAIL,
        "X-Tenant-Id": tid,
        "X-Roles": _SVC_ROLES,
        "X-Auth-Method": _SVC_AUTH_METHOD,
        "X-Service-Identity": _SVC_IDENTITY,
    }
    try:
        raw_key = get_settings().gateway_principal_hmac_key.get_secret_value()
    except Exception:  # pragma: no cover — key not configured
        raw_key = ""
    if raw_key:
        canonical = (
            f"{_SVC_USER_ID}\n{_SVC_EMAIL}\n{tid}\n{_SVC_ROLES}\n{_SVC_AUTH_METHOD}"
        ).encode()
        headers["X-Shielva-Principal-Signature"] = _hmac.new(
            raw_key.encode() if isinstance(raw_key, str) else raw_key,
            canonical,
            hashlib.sha256,
        ).hexdigest()
    return headers


class EntitlementGuard:
    """Per-pod entitlement enforcer with a short-TTL cache."""

    def __init__(self) -> None:
        # (tenant_id, feature_key) → (entitled, expiry_monotonic)
        self._cache: Dict[Tuple[str, str], Tuple[bool, float]] = {}

    # ── cache ──────────────────────────────────────────────────────────────
    def _cache_get(self, key: Tuple[str, str]) -> Optional[bool]:
        hit = self._cache.get(key)
        if hit is None:
            return None
        entitled, expiry = hit
        if time.monotonic() >= expiry:
            self._cache.pop(key, None)
            return None
        return entitled

    def _cache_set(self, key: Tuple[str, str], entitled: bool) -> None:
        ttl = max(1, get_settings().entitlement_cache_ttl_seconds)
        self._cache[key] = (entitled, time.monotonic() + ttl)

    # ── billing ────────────────────────────────────────────────────────────
    async def _ask_billing(self, tenant_id: str, feature_key: str) -> bool:
        from shielva_common.tls import internal_ca_verify

        s = get_settings()
        url = (
            f"{s.billing_url.rstrip('/')}"
            f"/v1/subscriptions/{tenant_id}/entitlements/{feature_key}"
        )
        async with httpx.AsyncClient(timeout=5.0, verify=internal_ca_verify()) as client:
            resp = await client.get(url, headers=_service_headers(tenant_id))
        if resp.status_code == 200:
            return bool(resp.json().get("entitled", False))
        # Any non-200 is a billing error → handled by the caller's fail mode.
        raise RuntimeError(f"billing returned HTTP {resp.status_code}: {resp.text[:200]}")

    async def is_entitled(self, tenant_id: str, feature_key: str) -> bool:
        key = (tenant_id, feature_key)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        try:
            entitled = await self._ask_billing(tenant_id, feature_key)
            self._cache_set(key, entitled)
            return entitled
        except Exception as exc:  # noqa: BLE001
            allow = not get_settings().entitlement_fail_closed
            logger.warning(
                "mcp.entitlement_check_failed",
                tenant_id=tenant_id, feature_key=feature_key,
                error=str(exc)[:200], decision="allow" if allow else "deny",
            )
            return allow  # fail open (default) or closed

    # ── enforcement ──────────────────────────────────────────────────────--
    async def enforce(self, *, principal: Principal, feature_key: str) -> None:
        s = get_settings()
        if not s.entitlement_enabled:
            return
        if principal.has_any_role(*_BYPASS_ROLES):
            return
        if await self.is_entitled(principal.tenant_id, feature_key):
            return
        logger.info(
            "mcp.entitlement_denied",
            tenant_id=principal.tenant_id, feature_key=feature_key,
        )
        raise HTTPException(
            status_code=403,
            detail={
                "error": "subscription_required",
                "feature": feature_key,
                "tenant_id": principal.tenant_id,
                "message": (
                    "This tenant is not entitled to this feature. Provision or "
                    "upgrade the subscription in shielva-billing."
                ),
            },
        )


_GUARD: Optional[EntitlementGuard] = None


def get_entitlement_guard() -> EntitlementGuard:
    global _GUARD
    if _GUARD is None:
        _GUARD = EntitlementGuard()
    return _GUARD


def require_entitlement(feature_key: str):
    """FastAPI dependency factory — gate a route on a tenant's entitlement to
    ``feature_key``. No-op when the gate is disabled.

    Usage:
        @router.post("/codegen/complete",
                     dependencies=[Depends(require_entitlement("codegen"))])
    """
    async def _dep(request: Request) -> None:
        if not get_settings().entitlement_enabled:
            return
        principal = require_principal(request)
        await get_entitlement_guard().enforce(principal=principal, feature_key=feature_key)

    return _dep


async def require_llm_entitlement(request: Request) -> None:
    """Ready-made dependency for LLM-serving routes — uses the configured LLM
    feature key (``entitlement_llm_feature_key``). No-op when the gate is
    disabled."""
    s = get_settings()
    if not s.entitlement_enabled:
        return
    principal = require_principal(request)
    await get_entitlement_guard().enforce(
        principal=principal, feature_key=s.entitlement_llm_feature_key,
    )
