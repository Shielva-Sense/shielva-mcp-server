"""PolicyEngine port."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .value_objects import FeatureDecision, PolicyDecision, PolicyRequest, QuotaDecision


class PolicyEngine(ABC):
    """Single port for RBAC + quota + feature decisions.

    Adapters MAY implement just the methods their backend supports —
    fallback adapters return permissive ``PolicyDecision(allowed=True)``
    for the unsupported methods. Tests use a fake that records every
    call and returns deterministic decisions.
    """

    @abstractmethod
    async def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        """RBAC + arbitrary policy. ``allowed=False`` MUST carry a
        ``reason`` for audit clarity."""

    @abstractmethod
    async def check_quota(
        self,
        *,
        tenant_id: str,
        resource_type: str,
        period: str = "monthly",
    ) -> QuotaDecision:
        """Has this tenant exceeded their quota for ``resource_type``
        in the current ``period``?"""

    @abstractmethod
    async def check_feature(
        self,
        *,
        tenant_id: str,
        feature: str,
    ) -> FeatureDecision:
        """Is ``feature`` enabled for this tenant? (Feature flags.)"""
