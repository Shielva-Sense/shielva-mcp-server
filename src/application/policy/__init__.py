"""Policy-context use cases.

* require_permission — wrap a permission check; raises PolicyDeniedError
                       on deny (so caller doesn't have to inspect the
                       PolicyDecision shape).
* check_quota        — read-through wrapper with audit logging.
* check_feature      — feature-flag read-through.
"""

from .services import PolicyApplicationService

__all__ = ["PolicyApplicationService"]
