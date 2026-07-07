"""PolicyApplicationService — require_permission raises on deny."""

from __future__ import annotations

import pytest

from src.application.policy import PolicyApplicationService
from src.domain.policy.errors import PolicyDeniedError
from src.domain.policy.repositories import PolicyEngine
from src.domain.policy.value_objects import (
    FeatureDecision,
    PolicyDecision,
    PolicyRequest,
    QuotaDecision,
)
from src.domain.shared.tenant import TenantContext


class _FakePolicy(PolicyEngine):
    def __init__(self, allow: bool = True, reason: str | None = None) -> None:
        self.allow = allow
        self.reason = reason
        self.calls: list[PolicyRequest] = []

    async def evaluate(self, request: PolicyRequest) -> PolicyDecision:
        self.calls.append(request)
        return PolicyDecision(allowed=self.allow, reason=self.reason)

    async def check_quota(self, *, tenant_id: str, resource_type: str, period: str = "monthly") -> QuotaDecision:
        return QuotaDecision(allowed=True, current=0, limit=1000, period=period)

    async def check_feature(self, *, tenant_id: str, feature: str) -> FeatureDecision:
        return FeatureDecision(enabled=True, reason=None)


def _tenant() -> TenantContext:
    return TenantContext(tenant_id="t1", user_id="u", user_email="u@x")


@pytest.mark.asyncio
async def test_require_permission_allowed():
    svc = PolicyApplicationService(engine=_FakePolicy(allow=True))
    decision = await svc.require_permission(
        tenant=_tenant(),
        action="read",
        resource_type="kb",
    )
    assert decision.allowed is True


@pytest.mark.asyncio
async def test_require_permission_denied_raises():
    svc = PolicyApplicationService(engine=_FakePolicy(allow=False, reason="role"))
    with pytest.raises(PolicyDeniedError) as exc_info:
        await svc.require_permission(
            tenant=_tenant(),
            action="delete",
            resource_type="kb",
        )
    assert "role" in str(exc_info.value)


@pytest.mark.asyncio
async def test_require_permission_passes_context_to_engine():
    fake = _FakePolicy(allow=True)
    svc = PolicyApplicationService(engine=fake)
    await svc.require_permission(
        tenant=_tenant(),
        action="query",
        resource_type="bot",
        resource_id="bot-123",
        metadata={"extra": "ctx"},
    )
    assert len(fake.calls) == 1
    req = fake.calls[0]
    assert req.action == "query"
    assert req.resource_id == "bot-123"
    assert req.metadata == {"extra": "ctx"}
