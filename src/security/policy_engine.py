"""
OPA Policy Engine Integration
Open Policy Agent integration for authorization and access control.
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import httpx
import structlog

logger = structlog.get_logger(__name__)


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
        # Normalize role for internal mapping (e.g., demo_user -> Demo User)
        role_map = {
            "demo_user": "Demo User",
            "tenant_admin": "Tenant Admin",
            "customer_admin": "Tenant Admin",
            "admin": "Tenant Admin",
            "bot_manager": "Bot Manager",
            "analyst": "Analyst",
            "viewer": "Viewer",
            "super_admin": "Super Admin"
        }
        context.user_role = role_map.get((context.user_role or "").lower(), context.user_role)

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
            logger.warning("OPA query failed, using fallback", error=str(e))
            
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
        # Normalize role
        role_map = {
            "demo_user": "Demo User",
            "tenant_admin": "Tenant Admin",
            "customer_admin": "Tenant Admin",
            "admin": "Tenant Admin",
            "bot_manager": "Bot Manager",
            "analyst": "Analyst",
            "viewer": "Viewer",
            "super_admin": "Super Admin"
        }
        user_role = role_map.get(user_role.lower(), user_role)

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
        # Normalize role
        role_map = {
            "demo_user": "Demo User",
            "tenant_admin": "Tenant Admin",
            "customer_admin": "Tenant Admin",
            "admin": "Tenant Admin",
            "bot_manager": "Bot Manager",
            "analyst": "Analyst",
            "viewer": "Viewer",
            "super_admin": "Super Admin"
        }
        user_role = role_map.get(user_role.lower(), user_role)

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
        """Default role-based permissions (Shielva Platform Roles)"""
        return {
            "Super Admin": {
                "bot": ["create", "read", "update", "delete", "deploy", "query"],
                "kb": ["create", "read", "update", "delete", "provision_kb"],
                "connector": ["create", "read", "update", "delete", "sync"],
                "user": ["create", "read", "update", "delete", "invite"],
                "settings": ["read", "update"]
            },
            "Tenant Admin": {
                "bot": ["create", "read", "update", "delete", "deploy", "query"],
                "kb": ["create", "read", "update", "delete", "provision_kb"],
                "connector": ["create", "read", "update", "delete", "sync"],
                "user": ["invite", "read"],
                "settings": ["read", "update"]
            },
            "Bot Manager": {
                "bot": ["create", "read", "update", "deploy", "query"],
                "kb": ["create", "read", "update", "provision_kb"],
                "connector": ["create", "read", "sync"],
                "user": ["read"],
                "settings": ["read"]
            },
            "Analyst": {
                "bot": ["read", "query"],
                "kb": ["read"],
                "connector": ["read"],
                "user": ["read"],
                "settings": ["read"]
            },
            "Viewer": {
                "bot": ["read", "query"],
                "kb": ["read"],
                "connector": ["read"],
                "user": ["read"],
                "settings": ["read"]
            },
            "Demo User": {
                "bot": ["read", "query", "deploy"],  # Can test existing bots
                "kb": ["read", "provision_kb"],
                "connector": [],
                "user": ["read"],
                "settings": ["read"]
            }
        }
    
    def _default_quotas(self) -> Dict[str, Dict[str, int]]:
        """Default resource quotas per role (Shielva Platform Quotas)"""
        return {
            "Super Admin": {"bot": 999, "kb": 999, "connector": 999},
            "Tenant Admin": {"bot": 50, "kb": 50, "connector": 25},
            "Bot Manager": {"bot": 10, "kb": 10, "connector": 5},
            "Analyst": {"bot": 0, "kb": 0, "connector": 0},
            "Viewer": {"bot": 0, "kb": 0, "connector": 0},
            "Demo User": {"bot": 1, "kb": 5, "connector": 0}
        }
    
    def _default_features(self) -> Dict[str, List[str]]:
        """Default features per role (Shielva Platform Features)"""
        return {
            "Super Admin": ["all"],
            "Tenant Admin": ["bot_deploy", "kb_advanced", "connector_all", "analytics_advanced"],
            "Bot Manager": ["bot_create", "kb_basic", "connector_limited"],
            "Analyst": ["analytics_basic"],
            "Viewer": ["read_only"],
            "Demo User": ["demo_access"]
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
