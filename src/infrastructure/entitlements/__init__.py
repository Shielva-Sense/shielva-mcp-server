"""Subscription entitlement enforcement for the LLM broker.

shielva-mcp is the single point every LLM call flows through, so it is the
correct (and only) place to enforce per-tenant subscription entitlements for
LLM features. See ``guard.py``.
"""

from .guard import (
    EntitlementGuard,
    get_entitlement_guard,
    require_entitlement,
    require_llm_entitlement,
)

__all__ = [
    "EntitlementGuard",
    "get_entitlement_guard",
    "require_entitlement",
    "require_llm_entitlement",
]
