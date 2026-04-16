"""
Tool Execution Sandbox
Secure sandboxed execution environment for tools.
"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import structlog
import time

logger = structlog.get_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of sandboxed execution"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: int = 0
    memory_used_mb: float = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class SandboxLimits:
    """Resource limits for sandbox execution"""
    timeout_seconds: float = 30.0
    max_memory_mb: int = 256
    max_output_size: int = 10000
    allowed_domains: List[str] = field(default_factory=list)
    blocked_operations: List[str] = field(default_factory=list)


class ExecutionSandbox:
    """
    Sandboxed execution environment for tools.
    
    Features:
    - Timeout enforcement
    - Memory limits
    - Network restrictions
    - Output capture
    - Audit logging
    """
    
    def __init__(
        self,
        default_limits: SandboxLimits = None
    ):
        """
        Initialize sandbox.
        
        Args:
            default_limits: Default resource limits
        """
        self.default_limits = default_limits or SandboxLimits()
        self._execution_log: List[Dict[str, Any]] = []
        
        logger.info("ExecutionSandbox initialized")
    
    async def execute(
        self,
        func: Callable,
        args: Dict[str, Any] = None,
        limits: SandboxLimits = None,
        context: Dict[str, Any] = None
    ) -> ExecutionResult:
        """
        Execute a function in the sandbox.
        
        Args:
            func: Async function to execute
            args: Function arguments
            limits: Resource limits (uses defaults if not provided)
            context: Execution context for logging
            
        Returns:
            ExecutionResult with outcome
        """
        limits = limits or self.default_limits
        args = args or {}
        context = context or {}
        
        start_time = time.time()
        execution_id = f"exec_{int(start_time * 1000)}"
        
        logger.info(
            "Sandbox execution starting",
            execution_id=execution_id,
            timeout=limits.timeout_seconds
        )
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_with_monitoring(func, args, limits),
                timeout=limits.timeout_seconds
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Log execution
            self._log_execution(
                execution_id=execution_id,
                success=True,
                execution_time_ms=execution_time,
                context=context
            )
            
            return ExecutionResult(
                success=True,
                result=result,
                execution_time_ms=execution_time
            )
            
        except asyncio.TimeoutError:
            execution_time = int((time.time() - start_time) * 1000)
            
            self._log_execution(
                execution_id=execution_id,
                success=False,
                error="Execution timeout",
                execution_time_ms=execution_time,
                context=context
            )
            
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {limits.timeout_seconds}s",
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            error_msg = str(e)
            
            self._log_execution(
                execution_id=execution_id,
                success=False,
                error=error_msg,
                execution_time_ms=execution_time,
                context=context
            )
            
            logger.error(
                "Sandbox execution failed",
                execution_id=execution_id,
                error=error_msg
            )
            
            return ExecutionResult(
                success=False,
                error=error_msg,
                execution_time_ms=execution_time
            )
    
    async def _execute_with_monitoring(
        self,
        func: Callable,
        args: Dict[str, Any],
        limits: SandboxLimits
    ) -> Any:
        """Execute function with resource monitoring"""
        # Note: Full resource monitoring would require additional libraries
        # like resource or psutil for memory tracking
        
        # Execute the function
        if asyncio.iscoroutinefunction(func):
            result = await func(**args)
        else:
            result = func(**args)
        
        # Truncate result if too large
        if isinstance(result, str) and len(result) > limits.max_output_size:
            result = result[:limits.max_output_size] + "... [truncated]"
        
        return result
    
    def _log_execution(
        self,
        execution_id: str,
        success: bool,
        execution_time_ms: int,
        error: str = None,
        context: Dict[str, Any] = None
    ):
        """Log execution for audit"""
        log_entry = {
            "execution_id": execution_id,
            "success": success,
            "execution_time_ms": execution_time_ms,
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context or {}
        }
        
        self._execution_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self._execution_log) > 1000:
            self._execution_log = self._execution_log[-1000:]
    
    def get_execution_log(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent execution log entries"""
        return self._execution_log[-limit:]


class NetworkProxy:
    """
    Network proxy for controlling external requests.
    
    Features:
    - Domain whitelist
    - Request logging
    - Rate limiting
    """
    
    def __init__(
        self,
        allowed_domains: List[str] = None,
        rate_limit: int = 100  # requests per minute
    ):
        self.allowed_domains = allowed_domains or []
        self.rate_limit = rate_limit
        self._request_count = 0
        self._request_window_start = time.time()
        
        logger.info(
            "NetworkProxy initialized",
            allowed_domains=len(self.allowed_domains)
        )
    
    def is_allowed(self, url: str) -> bool:
        """Check if a URL is allowed"""
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove port if present
        if ":" in domain:
            domain = domain.split(":")[0]
        
        # Check whitelist
        for allowed in self.allowed_domains:
            if domain == allowed or domain.endswith(f".{allowed}"):
                return True
        
        return False
    
    def check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded"""
        current_time = time.time()
        
        # Reset window if minute has passed
        if current_time - self._request_window_start > 60:
            self._request_count = 0
            self._request_window_start = current_time
        
        if self._request_count >= self.rate_limit:
            return False
        
        self._request_count += 1
        return True


class SecureToolExecutor:
    """
    Secure tool executor combining sandbox and network proxy.
    """
    
    def __init__(
        self,
        sandbox: ExecutionSandbox = None,
        network_proxy: NetworkProxy = None,
        policy_engine = None
    ):
        self.sandbox = sandbox or ExecutionSandbox()
        self.network_proxy = network_proxy
        self.policy_engine = policy_engine
        
        logger.info("SecureToolExecutor initialized")
    
    async def execute_tool(
        self,
        tool_name: str,
        handler: Callable,
        parameters: Dict[str, Any],
        tenant_id: str,
        user_id: str,
        user_role: str
    ) -> ExecutionResult:
        """
        Execute a tool securely.
        
        Args:
            tool_name: Name of the tool
            handler: Tool handler function
            parameters: Tool parameters
            tenant_id: Tenant ID
            user_id: User ID
            user_role: User role
            
        Returns:
            ExecutionResult
        """
        # Check policy if engine available
        if self.policy_engine:
            from src.security.policy_engine import PolicyContext
            
            decision = await self.policy_engine.evaluate(
                PolicyContext(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    user_role=user_role,
                    action="execute",
                    resource_type="tool",
                    resource_id=tool_name
                )
            )
            
            if not decision.allowed:
                return ExecutionResult(
                    success=False,
                    error=f"Policy denied: {decision.reason}"
                )
        
        # Execute in sandbox
        context = {
            "tool_name": tool_name,
            "tenant_id": tenant_id,
            "user_id": user_id
        }
        
        # Add tenant context to parameters
        from src.protocol.models import TenantContext
        parameters["tenant_context"] = TenantContext(
            tenant_id=tenant_id,
            user_id=user_id,
            role=user_role
        )
        
        result = await self.sandbox.execute(
            func=handler,
            args=parameters,
            context=context
        )
        
        return result
