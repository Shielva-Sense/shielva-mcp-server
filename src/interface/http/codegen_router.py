"""
Codegen completion route — internal API for integration-builder service.

Routes code-generation prompts through MCP's LiteLLM router (with its
configured model, API key, and fallback chain) without running the full
bot / KB / RAG pipeline.

Endpoint: POST /mcp/v1/codegen/complete
Caller:   integration-builder (shielva-connectors) when LLM_MODE=mcp

Security: Requires X-Tenant-ID header (same as every MCP endpoint).
          Skips the MongoDB permission lookup that user-facing queries do —
          this is an internal service-to-service call.
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, List, Optional

from src.protocol.models import TenantContext

logger = structlog.get_logger(__name__)

codegen_router = APIRouter(prefix="/mcp/v1/codegen", tags=["codegen-internal"])


# ── Request / Response models ─────────────────────────────────────────

class CodegenCompleteRequest(BaseModel):
    """Request body for code-generation completion."""

    messages: List[Dict[str, str]]
    """OpenAI-format message list, e.g. [{"role": "user", "content": "..."}]."""

    system: str = ""
    """System prompt injected before the message list."""

    max_tokens: int = 8192
    temperature: float = 0.3
    model: Optional[str] = None
    """Optional model override, e.g. 'gemini/gemini-2.5-pro'. Defaults to MCP's configured model."""


class CodegenCompleteResponse(BaseModel):
    """Response from code-generation completion."""

    text: str
    """Generated text (Python code, JSON, etc.)."""

    model: str
    """Model that produced the response."""

    tokens_used: int
    """Total tokens consumed (0 when not available, e.g. streaming path)."""


# ── Dependency: lightweight tenant extraction ─────────────────────────

def _extract_tenant(request: Request) -> TenantContext:
    """
    Extract TenantContext from request headers for internal calls.

    Skips the MongoDB permission lookup used by user-facing query endpoints —
    integration-builder is a trusted internal service and tenant_id is
    the only isolation key needed here.
    """
    tenant_id = request.headers.get("X-Tenant-ID", "").strip()
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Missing X-Tenant-ID header")

    return TenantContext(
        tenant_id=tenant_id,
        user_id=request.headers.get("X-User-ID", "integration-builder"),
        user_email=request.headers.get("X-User-Email", "internal@shielva.ai"),
        role=request.headers.get("X-User-Role", "Customer_Basic"),
        permissions=[],  # Internal call — no permission gating on codegen
    )


# ── Route ─────────────────────────────────────────────────────────────

@codegen_router.post("/complete", response_model=CodegenCompleteResponse)
async def codegen_complete(
    body: CodegenCompleteRequest,
    request: Request,
    tenant_context: TenantContext = Depends(_extract_tenant),
) -> CodegenCompleteResponse:
    """
    Route a code-generation prompt through MCP's LiteLLM router.

    Called by integration-builder (shielva-connectors) when
    INTEGRATION_LLM_MODE=mcp.  Uses MCP's configured model + fallback
    chain — no API keys needed in the integration service.

    The system prompt is injected as the first message with role=system
    so it works across all LiteLLM-supported providers.
    """
    llm_router = getattr(request.app.state, "llm_router", None)
    if llm_router is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "MCP LLM router is not yet initialized. "
                "The MCP server may still be starting up."
            ),
        )

    # Build the messages list: system prompt first (if any), then conversation turns
    msgs: List[Dict[str, str]] = []
    if body.system:
        msgs.append({"role": "system", "content": body.system})
    msgs.extend(body.messages)

    logger.info(
        "codegen.complete",
        tenant_id=tenant_context.tenant_id,
        model_override=body.model,
        msg_count=len(msgs),
        system_length=len(body.system),
        max_tokens=body.max_tokens,
    )

    try:
        response = await llm_router.execute(
            messages=msgs,
            tools=None,          # No tool-calling for direct codegen
            tenant_context=tenant_context,
            stream=False,
            model=body.model,    # None → uses MCP's configured default model
        )
    except Exception as exc:
        logger.error(
            "codegen.complete_failed",
            tenant_id=tenant_context.tenant_id,
            error=str(exc),
        )
        # Sanitize: don't leak internal paths/keys in error responses
        safe_detail = f"Code generation failed: {type(exc).__name__}"
        raise HTTPException(status_code=500, detail=safe_detail) from exc

    logger.info(
        "codegen.complete_ok",
        tenant_id=tenant_context.tenant_id,
        model=response.model,
        tokens_used=response.tokens_used,
        response_length=len(response.answer),
    )

    return CodegenCompleteResponse(
        text=response.answer,
        model=response.model,
        tokens_used=response.tokens_used,
    )


# ── Fix-Agent endpoint (uses MCP tool-calling loop) ───────────────────

class CodegenFixAgentRequest(BaseModel):
    """Request body for agent-driven code fix."""

    broken_code: str
    """The Python code that failed."""

    error_output: str
    """Full pytest / Python error output string."""

    connector_class: str = ""
    """Expected connector class name, e.g. 'GmailConnector'."""

    user_prompt: str = ""
    """Original user intent (e.g. 'Build a Gmail connector that sends emails')."""

    step_memory_summary: str = ""
    """Summary of what previous steps produced (installed packages, etc.)."""

    max_tokens: int = 16384
    temperature: float = 0.2
    model: Optional[str] = None


class CodegenFixAgentResponse(BaseModel):
    fixed_code: str
    """The corrected Python code."""

    fix_explanation: str = ""
    """Brief explanation of what was fixed."""

    tools_called: List[str] = []
    """Names of MCP tools called during the fix cycle."""

    model: str = ""
    tokens_used: int = 0


@codegen_router.post("/fix-agent", response_model=CodegenFixAgentResponse)
async def codegen_fix_agent(
    body: CodegenFixAgentRequest,
    request: Request,
    tenant_context: TenantContext = Depends(_extract_tenant),
) -> CodegenFixAgentResponse:
    """
    Use MCP's tool-calling agent loop to intelligently fix broken connector code.

    The agent:
      1. Calls codegen_categorize_error to classify the failure
      2. Calls codegen_check_pytest_structure or codegen_analyze_imports based on category
      3. Generates a targeted fix using the gathered intelligence
      4. Validates the fix with codegen_validate_python before returning

    This gives smarter, more targeted fixes than a single LLM call with just the error string.
    """
    llm_router = getattr(request.app.state, "llm_router", None)
    tool_registry = getattr(request.app.state, "tool_registry", None)

    if llm_router is None:
        raise HTTPException(status_code=503, detail="MCP LLM router not initialized")

    # Build system prompt for the fix agent
    system = (
        "You are an expert Python connector developer for the Shielva Integration Builder.\n"
        "Your job is to fix broken Python connector or test code by:\n"
        "1. Using codegen_categorize_error to classify the error\n"
        "2. Using codegen_check_pytest_structure or codegen_analyze_imports to find structural issues\n"
        "3. Using codegen_validate_python to verify your fix is syntactically correct\n"
        "4. Returning ONLY the complete fixed Python source code — no markdown fences, no explanations.\n\n"
        f"Connector class name: {body.connector_class or 'unknown'}\n"
        f"User intent: {body.user_prompt or 'build a connector'}\n"
        f"Context from previous steps: {body.step_memory_summary or 'none'}\n\n"
        "CRITICAL: Your final response must be ONLY valid Python code starting with imports or class definition."
    )

    user_message = (
        f"Fix this broken Python code.\n\n"
        f"## Error Output\n```\n{body.error_output[:3000]}\n```\n\n"
        f"## Broken Code\n```python\n{body.broken_code[:8000]}\n```\n\n"
        "Use the available tools to diagnose the error category and structural issues, "
        "then return the complete fixed Python code."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    # Get codegen tools for the agent
    tools = []
    if tool_registry is not None:
        # Fetch just the codegen tools (not rag_query etc.)
        codegen_tool_names = {
            "codegen_validate_python",
            "codegen_analyze_imports",
            "codegen_categorize_error",
            "codegen_check_pytest_structure",
        }
        all_tools = tool_registry.get_all_tools()
        from src.routing.llm_router import ToolSpec
        for t in all_tools:
            if t.name in codegen_tool_names:
                registered = tool_registry._tools.get(t.name)
                if registered:
                    tools.append(ToolSpec(
                        name=t.name,
                        description=t.description,
                        parameters=tool_registry._build_parameters_schema(t),
                        handler=registered.handler,
                    ))

    logger.info(
        "codegen.fix_agent",
        tenant_id=tenant_context.tenant_id,
        model_override=body.model,
        connector_class=body.connector_class,
        error_preview=body.error_output[:100],
        tools_available=len(tools),
    )

    try:
        response = await llm_router.execute(
            messages=messages,
            tools=tools or None,
            tenant_context=tenant_context,
            stream=False,
            model=body.model,
        )
    except Exception as exc:
        logger.error("codegen.fix_agent_failed", error=str(exc), tenant_id=tenant_context.tenant_id)
        safe_detail = f"Fix agent failed: {type(exc).__name__}"
        raise HTTPException(status_code=500, detail=safe_detail) from exc

    fixed_code = response.answer.strip()
    # Strip markdown fences if the LLM added them (handles ```python, ``` python, etc.)
    if fixed_code.startswith("```"):
        lines = fixed_code.split("\n")
        # First line is the opening fence (```python, ```py, ``` etc.) — always remove it
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Last line is the closing fence — remove it
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        fixed_code = "\n".join(lines)

    tools_called = [tc.tool_name for tc in (response.tool_calls or [])]

    logger.info(
        "codegen.fix_agent_ok",
        tenant_id=tenant_context.tenant_id,
        model=response.model,
        tokens_used=response.tokens_used,
        tools_called=tools_called,
        fixed_code_length=len(fixed_code),
    )

    return CodegenFixAgentResponse(
        fixed_code=fixed_code,
        fix_explanation=f"Fixed via MCP agent (tools used: {', '.join(tools_called) or 'none'})",
        tools_called=tools_called,
        model=response.model,
        tokens_used=response.tokens_used,
    )
