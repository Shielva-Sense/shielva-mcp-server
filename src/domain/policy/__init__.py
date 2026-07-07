"""Policy bounded context.

Wraps Shielva's OPA-backed RBAC + quota + feature-flag system
behind a clean port. The legacy ``OPAPolicyEngine`` mixes three
distinct concerns (RBAC, quotas, features); we expose a single
``PolicyEngine`` port with three methods so adapters can keep them
co-located while domain code consumes the right one for each
question.

Public surface:
    Value objects : PolicyDecision, PolicyRequest, QuotaDecision,
                    FeatureDecision
    Ports         : PolicyEngine
    Errors        : PolicyDeniedError
"""

from .errors import PolicyDeniedError
from .repositories import PolicyEngine
from .value_objects import (
    FeatureDecision,
    PolicyDecision,
    PolicyRequest,
    QuotaDecision,
)

__all__ = [
    "FeatureDecision",
    "PolicyDecision",
    "PolicyDeniedError",
    "PolicyEngine",
    "PolicyRequest",
    "QuotaDecision",
]
