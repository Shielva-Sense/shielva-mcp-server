"""
OPA Policy Engine Integration
Open Policy Agent integration for authorization and access control.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import httpx
import structlog

logger = structlog.get_logger(__name__)


# ── Canonical role normalization (single source of truth) ───────────────────────
# The IdP (shielva-identity) defines exactly THREE canonical roles. This MUST stay in
# lockstep with shielva-identity/app/core/roles.py (UserRole + LEGACY_ROLE_MAP +
# normalise_role):
#
#   platform_owner — internal Shielva founder; all-access, plan-exempt.
#   tenant_admin   — tenant administrator; full business-app access.
#   developer      — standard end-user; core apps (arc/acp/tms), no destructive ops.
#
# Legacy/OIDC role strings normalise to one of the three; anything unknown → developer
# (the IdP default — never a hard deny, so a deployed bot stays chattable). External
# bot end-users carry no platform role and therefore also resolve to developer, which
# holds bot:query. Previously the fallback used invented display-name roles and omitted
# platform_owner entirely → legitimate principals were denied with "Unknown role".
_CANONICAL_ROLES = ("platform_owner", "tenant_admin", "developer")

_LEGACY_ROLE_MAP: Dict[str, str] = {
    "super_admin": "platform_owner",
    "superadmin":  "platform_owner",
    "admin":       "tenant_admin",
    "bot_manager": "developer",
    "analyst":     "developer",
    "viewer":      "developer",
    "partner":     "developer",
}

# Most-privileged wins when a principal carries multiple roles.
_ROLE_PRIORITY = ("platform_owner", "tenant_admin", "developer")
_DEFAULT_POLICY_ROLE = "developer"


def normalize_role(raw_role: Optional[str]) -> str:
    """Map a principal's raw role string to one of the three canonical roles.

    Mirrors shielva-identity's ``normalise_role``: case-insensitive, legacy-string aware,
    and multi-role aware (comma-joined principals resolve to the most-privileged). Unknown
    roles default to ``developer``.
    """
    if not raw_role:
        return _DEFAULT_POLICY_ROLE
    matched: List[str] = []
    for tok in str(raw_role).replace(";", ",").split(","):
        r = tok.strip().lower().replace(" ", "_").replace("-", "_")
        if not r:
            continue
        r = _LEGACY_ROLE_MAP.get(r, r)
        if r in _CANONICAL_ROLES:
            matched.append(r)
    for role in _ROLE_PRIORITY:  # most-privileged wins
        if role in matched:
            return role
    return _DEFAULT_POLICY_ROLE


@dataclass
class PolicyDecision:
    """Result of a policy evaluation"""
    allowed: bool
    reason: Optional[str] = None
    conditions: Dict[str, Any] = None


@dataclass
class PolicyContext:
    """Context for policy evaluation"""
    tenant_id: str
    user_id: str
    user_role: str
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    metadata: Dict[str, Any] = None


class OPAPolicyEngine:
    """
    OPA (Open Policy Agent) integration for authorization.
    
    Handles:
    - RBAC (Role-Based Access Control)
    - Resource quotas
    - Feature flags
    - Data filtering policies
    
    Policies are defined in Rego and loaded into OPA.
    """
    
    def __init__(
        self,
        opa_url: str = "http://localhost:8181",
        policy_path: str = "/v1/data/shielva"
    ):
        """
        Initialize OPA policy engine.
        
        Args:
            opa_url: OPA server URL
            policy_path: Base path for Shielva policies
        """
        self.opa_url = opa_url
        self.policy_path = policy_path
        self._http_client = httpx.AsyncClient(timeout=10.0)
        
        # Fallback policies for when OPA is unavailable
        self._fallback_enabled = True
        self._role_permissions = self._default_role_permissions()
        # When OPA isn't deployed, every request would otherwise log a warning. Warn
        # once, then stay quiet (debug) so the fallback path doesn't spam prod logs.
        self._opa_warned = False

        logger.info("OPAPolicyEngine initialized", opa_url=opa_url)
    
    async def evaluate(
        self,
        context: PolicyContext
    ) -> PolicyDecision:
        """
        Evaluate a policy.
        
        Args:
            context: Policy evaluation context
            
        Returns:
            PolicyDecision with allow/deny result
        """
        # Normalize role to a canonical policy role (see normalize_role).
        context.user_role = normalize_role(context.user_role)

        logger.debug(
            "Evaluating policy",
            action=context.action,
            resource=context.resource_type,
            role=context.user_role
        )
        
        try:
            # Build OPA input
            input_data = {
                "tenant_id": context.tenant_id,
                "user_id": context.user_id,
                "role": context.user_role,
                "action": context.action,
                "resource": {
                    "type": context.resource_type,
                    "id": context.resource_id
                },
                "metadata": context.metadata or {}
            }
            
            # Query OPA
            result = await self._query_opa(
                f"{self.policy_path}/authz/allow",
                input_data
            )
            
            return PolicyDecision(
                allowed=result.get("result", False),
                reason=result.get("reason"),
                conditions=result.get("conditions")
            )
            
        except Exception as e:
            # Warn once (then debug) — when OPA isn't deployed this fires on every
            # request, and the fallback RBAC is the intended authority in that mode.
            if not self._opa_warned:
                logger.warning("OPA unavailable — using fallback RBAC for this and subsequent requests", error=str(e))
                self._opa_warned = True
            else:
                logger.debug("OPA query failed, using fallback", error=str(e))

            if self._fallback_enabled:
                return self._fallback_evaluate(context)

            # Fail closed
            return PolicyDecision(
                allowed=False,
                reason=f"Policy evaluation failed: {str(e)}"
            )
    
    async def check_quota(
        self,
        tenant_id: str,
        user_id: str,
        user_role: str,
        resource_type: str,
        current_count: int
    ) -> PolicyDecision:
        """
        Check if a quota allows a resource creation.
        
        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            user_role: User role
            resource_type: Type of resource (kb, bot, connector)
            current_count: Current count of resources
            
        Returns:
            PolicyDecision indicating if quota allows creation
        """
        # Normalize role to a canonical policy role (see normalize_role).
        user_role = normalize_role(user_role)

        try:
            input_data = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "role": user_role,
                "resource_type": resource_type,
                "current_count": current_count
            }
            
            result = await self._query_opa(
                f"{self.policy_path}/quota/allow",
                input_data
            )
            
            return PolicyDecision(
                allowed=result.get("result", False),
                reason=result.get("reason"),
                conditions={
                    "limit": result.get("limit"),
                    "remaining": result.get("remaining")
                }
            )
            
        except Exception as e:
            logger.warning("Quota check failed, using fallback", error=str(e))
            return self._fallback_quota_check(user_role, resource_type, current_count)
    
    async def get_data_filters(
        self,
        tenant_id: str,
        user_id: str,
        user_role: str,
        resource_type: str
    ) -> Dict[str, Any]:
        """
        Get data filtering rules.
        
        Returns filter conditions to apply to queries.
        """
        try:
            input_data = {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "role": user_role,
                "resource_type": resource_type
            }
            
            result = await self._query_opa(
                f"{self.policy_path}/filter/data",
                input_data
            )
            
            return result.get("result", {})
            
        except Exception as e:
            logger.warning("Filter query failed", error=str(e))
            return {"tenant_id": tenant_id}  # Always filter by tenant
    
    async def check_feature(
        self,
        tenant_id: str,
        user_role: str,
        feature: str
    ) -> bool:
        """
        Check if a feature is enabled for a user/tenant.
        
        Args:
            tenant_id: Tenant identifier
            user_role: User role
            feature: Feature name
            
        Returns:
            True if feature is enabled
        """
        # Normalize role to a canonical policy role (see normalize_role).
        user_role = normalize_role(user_role)

        try:
            input_data = {
                "tenant_id": tenant_id,
                "role": user_role,
                "feature": feature
            }
            
            result = await self._query_opa(
                f"{self.policy_path}/features/enabled",
                input_data
            )
            
            return result.get("result", False)
            
        except Exception as e:
            logger.warning("Feature check failed", error=str(e))
            return self._fallback_feature_check(user_role, feature)
    
    async def _query_opa(
        self,
        path: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query OPA endpoint"""
        url = f"{self.opa_url}{path}"
        
        response = await self._http_client.post(
            url,
            json={"input": input_data}
        )
        response.raise_for_status()
        
        return response.json()
    
    # ===== Fallback Policies =====
    
    def _default_role_permissions(self) -> Dict[str, Dict[str, List[str]]]:
        """Default role-based permissions, keyed by the three canonical IdP roles
        (see normalize_role / shielva-identity roles.py). Bot resource actions include
        query/test_bot/provision_bot (the chat + test surface) — all three roles can
        chat/test a bot; only destructive ops differ."""
        return {
            "platform_owner": {
                "bot": ["create", "read", "update", "delete", "deploy", "query", "test_bot", "provision_bot"],
                "kb": ["create", "read", "update", "delete", "provision_kb"],
                "connector": ["create", "read", "update", "delete", "sync"],
                "user": ["create", "read", "update", "delete", "invite"],
                "settings": ["read", "update"]
            },
            "tenant_admin": {
                "bot": ["create", "read", "update", "delete", "deploy", "query", "test_bot", "provision_bot"],
                "kb": ["create", "read", "update", "delete", "provision_kb"],
                "connector": ["create", "read", "update", "delete", "sync"],
                "user": ["invite", "read"],
                "settings": ["read", "update"]
            },
            "developer": {
                # Core apps: create/manage and TEST/chat bots, but no destructive ops
                # (no delete) — mirrors shielva-identity _DEVELOPER_DENY.
                "bot": ["create", "read", "update", "deploy", "query", "test_bot", "provision_bot"],
                "kb": ["create", "read", "update", "provision_kb"],
                "connector": ["read"],
                "user": ["read"],
                "settings": ["read"]
            }
        }

    def _default_quotas(self) -> Dict[str, Dict[str, int]]:
        """Default resource quotas per canonical role (gates CREATE only; query is unmetered)."""
        return {
            "platform_owner": {"bot": 999, "kb": 999, "connector": 999},
            "tenant_admin": {"bot": 999, "kb": 999, "connector": 999},
            "developer": {"bot": 50, "kb": 50, "connector": 10}
        }

    def _default_features(self) -> Dict[str, List[str]]:
        """Default features per canonical role."""
        return {
            "platform_owner": ["all"],
            "tenant_admin": ["all"],
            "developer": ["bot_create", "kb_basic", "chat"]
        }
    
    def _fallback_evaluate(self, context: PolicyContext) -> PolicyDecision:
        """Fallback policy evaluation"""
        role = context.user_role
        resource_type = context.resource_type
        action = context.action
        
        if role not in self._role_permissions:
            return PolicyDecision(
                allowed=False,
                reason=f"Unknown role: {role}"
            )
        
        permissions = self._role_permissions[role]
        
        if resource_type not in permissions:
            return PolicyDecision(
                allowed=False,
                reason=f"No access to resource type: {resource_type}"
            )
        
        allowed_actions = permissions[resource_type]
        
        if action in allowed_actions:
            return PolicyDecision(allowed=True)
        
        return PolicyDecision(
            allowed=False,
            reason=f"Action '{action}' not allowed for role '{role}' on '{resource_type}'"
        )
    
    def _fallback_quota_check(
        self,
        user_role: str,
        resource_type: str,
        current_count: int
    ) -> PolicyDecision:
        """Fallback quota check"""
        quotas = self._default_quotas()
        
        if user_role not in quotas:
            return PolicyDecision(
                allowed=False,
                reason=f"Unknown role: {user_role}"
            )
        
        role_quotas = quotas[user_role]
        limit = role_quotas.get(resource_type, 0)
        
        if current_count < limit:
            return PolicyDecision(
                allowed=True,
                conditions={
                    "limit": limit,
                    "remaining": limit - current_count - 1
                }
            )
        
        return PolicyDecision(
            allowed=False,
            reason=f"Quota exceeded: {current_count}/{limit} {resource_type}s",
            conditions={
                "limit": limit,
                "remaining": 0
            }
        )
    
    def _fallback_feature_check(
        self,
        user_role: str,
        feature: str
    ) -> bool:
        """Fallback feature check"""
        features = self._default_features()
        
        if user_role not in features:
            return False
        
        role_features = features[user_role]
        
        return "all" in role_features or feature in role_features
    
    async def close(self):
        """Close HTTP client"""
        await self._http_client.aclose()


# ===== Rego Policies (Reference) =====

SAMPLE_REGO_POLICIES = '''
# Shielva Authorization Policies

package shielva.authz

default allow = false

# Admin can do anything
allow {
    input.role == "Admin"
}

# Role-based access
allow {
    role_permissions[input.role][input.resource.type][_] == input.action
}

# Role permissions mapping
role_permissions := {
    "Customer_Ultra": {
        "bot": ["create", "read", "update", "delete", "deploy"],
        "kb": ["create", "read", "update", "delete"],
        "connector": ["create", "read", "update", "delete", "sync"]
    },
    "Customer_Premium": {
        "bot": ["create", "read", "update", "deploy"],
        "kb": ["create", "read", "update"],
        "connector": ["create", "read", "sync"]
    },
    "Customer_Essential": {
        "bot": ["create", "read"],
        "kb": ["create", "read"],
        "connector": ["read"]
    },
    "Customer_Basic": {
        "bot": ["read"]
    }
}

# Quota checking
package shielva.quota

default allow = false

allow {
    limit := quotas[input.role][input.resource_type]
    input.current_count < limit
}

quotas := {
    "Customer_Ultra": {"bot": 50, "kb": 50, "connector": 25},
    "Customer_Premium": {"bot": 20, "kb": 20, "connector": 10},
    "Customer_Essential": {"bot": 3, "kb": 3, "connector": 2},
    "Customer_Basic": {"bot": 0, "kb": 0, "connector": 0}
}

# Feature flags
package shielva.features

default enabled = false

enabled {
    input.role == "Admin"
}

enabled {
    feature_list[input.role][_] == input.feature
}

feature_list := {
    "Customer_Ultra": ["bot_deploy", "kb_advanced", "analytics_advanced", "api_access"],
    "Customer_Premium": ["bot_deploy", "analytics_basic"],
    "Customer_Essential": ["demo_access"],
    "Customer_Basic": ["demo_access"]
}

# Data filtering
package shielva.filter

data = filters {
    # Always filter by tenant
    filters := {"tenant_id": input.tenant_id}
}
'''
