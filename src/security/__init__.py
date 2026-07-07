"""
Shielva MCP Server Security Layer
Authorization, policy enforcement, and sandboxed execution.
"""

from .policy_engine import (
    OPAPolicyEngine,
    PolicyContext,
    PolicyDecision,
    normalize_role,
)
from .sandbox import (
    ExecutionResult,
    ExecutionSandbox,
    SandboxLimits,
    SecureToolExecutor,
)

__all__ = [
    "ExecutionResult",
    "ExecutionSandbox",
    "OPAPolicyEngine",
    "PolicyContext",
    "PolicyDecision",
    "SandboxLimits",
    "SecureToolExecutor",
    "normalize_role",
]
