"""Adapter wrapping the existing ``OPAPolicyEngine`` into the new
:class:`domain.policy.PolicyEngine` port.

The legacy engine returns its own ``PolicyDecision`` dataclass; we
translate to the domain VO so application code only depends on the
domain types. Quota + feature passthroughs mirror the legacy
fallback behaviour when OPA is unreachable.
"""
from __future__ import annotations

from typing import Any

import structlog

from src.domain.policy.repositories import PolicyEngine
from src.domain.policy.value_objects import (
    FeatureDecision, PolicyDecision, PolicyRequest, QuotaDecision,
)

logger = structlog.get_logger(__name__)


class OPAPolicyEngineAdapter(PolicyEngine):
    def __init__(self, legacy_engine: Any) -> None:
        self._engine = legacy_engine

    async def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        # Lazy import so the domain stays clean of the legacy types.
        from src.security.policy_engine import PolicyContext as LegacyContext
        legacy_ctx = LegacyContext(
            tenant_id     = request.tenant_id,
            user_id       = request.user_id,
            user_role     = request.user_role,
            action        = request.action,
            resource_type = request.resource_type,
            resource_id   = request.resource_id,
            metadata      = dict(request.metadata),
        )
        legacy_decision = await self._engine.evaluate(legacy_ctx)
        return PolicyDecision(
            allowed    = bool(legacy_decision.allowed),
            reason     = legacy_decision.reason,
            conditions = dict(legacy_decision.conditions or {}),
        )

    async def check_quota(
        self, *, tenant_id: str, resource_type: str, period: str = "monthly",
    ) -> QuotaDecision:
        # Legacy signature: check_quota(tenant_id, resource_type, period).
        result = await self._engine.check_quota(
            tenant_id=tenant_id, resource_type=resource_type, period=period,
        )
        # The legacy engine returns a dict (per the fallback path we
        # observed). Adapt defensively.
        if isinstance(result, dict):
            return QuotaDecision(
                allowed = bool(result.get("allowed", True)),
                current = int(result.get("current", 0) or 0),
                limit   = int(result.get("limit",   0) or 0),
                period  = str(result.get("period",  period)),
            )
        # Or it may return its own dataclass — duck-type the fields.
        return QuotaDecision(
            allowed = bool(getattr(result, "allowed", True)),
            current = int(getattr(result, "current", 0) or 0),
            limit   = int(getattr(result, "limit", 0) or 0),
            period  = str(getattr(result, "period", period)),
        )

    async def check_feature(self, *, tenant_id: str, feature: str
                            ) -> FeatureDecision:
        result = await self._engine.check_feature(
            tenant_id=tenant_id, feature=feature,
        )
        if isinstance(result, bool):
            return FeatureDecision(enabled=result, reason=None)
        if isinstance(result, dict):
            return FeatureDecision(
                enabled = bool(result.get("enabled", False)),
                reason  = result.get("reason"),
            )
        return FeatureDecision(
            enabled = bool(getattr(result, "enabled", False)),
            reason  = getattr(result, "reason", None),
        )
