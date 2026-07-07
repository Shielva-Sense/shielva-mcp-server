"""Policy-context value objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PolicyRequest:
    """Question we're asking the engine.

    ``action`` + ``resource_type`` follow the canonical RBAC pair
    (e.g. ``query`` on ``bot``, ``provision`` on ``kb``). The full
    list is the OPA policy bundle's concern.
    """

    tenant_id: str
    user_id: str
    user_role: str
    action: str
    resource_type: str
    resource_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: str | None = None
    conditions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QuotaDecision:
    allowed: bool
    current: int
    limit: int
    period: str = ""  # e.g. "daily" / "monthly"


@dataclass(frozen=True, slots=True)
class FeatureDecision:
    """For feature-flag style on/off checks (e.g. is 'fix-agent'
    enabled for this tenant)."""

    enabled: bool
    reason: str | None = None
