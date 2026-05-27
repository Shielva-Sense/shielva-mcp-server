"""CompleteWithToolLoop — generic OpenAI-style tool-calling loop.

Composes :class:`LLMProvider` (one-turn completion) with
:class:`ToolExecutor` (one tool execution) into the multi-turn
loop every modern function-calling LLM needs:

    1. Send (system+user+history+tools) to the provider.
    2. If response.tool_calls is non-empty:
         a. For each call: look up the tool via ToolCatalogue,
            execute via ToolExecutor, capture the result.
         b. Append the assistant's tool_calls turn + one role=tool
            message per call.
         c. Go to (1).
    3. Otherwise: response.content IS the final assistant turn —
       return it.

Cap at ``max_iterations`` so a misbehaving model can't burn provider
quota in an infinite loop. After the cap we force one last turn
*without tools* so the model has to summarise rather than emit
another tool call.

Why this lives in ``application/llm`` (and not ``application/chat``):
    Tool-calling is an LLM-shaped capability that other use cases
    reuse (codegen fix-agent, HandleQuery's future decomposition,
    integration-builder fan-out). The chat context owns *sessions*;
    this owns the *loop*. The two compose at the application layer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import structlog

from src.domain.llm.repositories import LLMProvider
from src.domain.llm.value_objects import (
    FinishReason, LLMMessage, LLMRequest, LLMResponse, LLMToolCall,
    MessageRole, ModelId,
)
from src.domain.shared.tenant import TenantContext
from src.domain.tools.repositories import ToolCatalogue, ToolExecutor
from src.domain.tools.value_objects import ToolName

logger = structlog.get_logger(__name__)


# ── Audit shape ────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ExecutedToolCall:
    """Audit record of one tool the loop executed."""
    name:      str
    arguments: Dict[str, Any]
    result:    str        # serialised content from ToolResult
    is_error:  bool


@dataclass(frozen=True, slots=True)
class ToolLoopInput:
    messages:        Tuple[LLMMessage, ...]
    tools:           Tuple[Dict[str, Any], ...]    # OpenAI tool schemas
    model:           Optional[ModelId] = None
    max_tokens:      int               = 1024
    temperature:     float             = 0.1
    max_iterations:  int               = 5
    tool_choice:     Optional[Any]     = None      # default: "auto" when tools present


@dataclass(frozen=True, slots=True)
class ToolLoopOutput:
    answer:        str
    executed:      Tuple[ExecutedToolCall, ...]
    model:         ModelId
    finish_reason: FinishReason
    tokens_used:   int
    iterations:    int
    truncated:     bool      # True when we hit max_iterations


class CompleteWithToolLoopUseCase:
    """Drive a tool-calling loop against the configured LLM provider.

    The use case owns *no state* across calls — every invocation is
    independent. Concurrent calls are safe.
    """

    def __init__(
        self,
        *,
        provider:        LLMProvider,
        catalogue:       ToolCatalogue,
        executor:        ToolExecutor,
    ) -> None:
        self._provider  = provider
        self._catalogue = catalogue
        self._executor  = executor

    async def execute(
        self,
        *,
        input_: ToolLoopInput,
        tenant: TenantContext,
    ) -> ToolLoopOutput:
        started = time.monotonic()
        history: List[LLMMessage] = list(input_.messages)
        executed: List[ExecutedToolCall] = []
        last_model = ModelId("")
        last_finish = FinishReason.OTHER
        total_tokens = 0
        iteration = 0

        logger.info(
            "mcp.tool_loop_start",
            tenant_id=tenant.tenant_id,
            tool_count=len(input_.tools),
            max_iterations=input_.max_iterations,
        )

        for iteration in range(1, input_.max_iterations + 1):
            response = await self._call_provider(
                history, input_, tenant, use_tools=True,
            )
            last_model = response.model
            last_finish = response.finish_reason
            total_tokens += response.usage.total_tokens

            if not response.tool_calls:
                # Plain text answer — we're done.
                logger.info(
                    "mcp.tool_loop_ok",
                    tenant_id=tenant.tenant_id,
                    iterations=iteration,
                    executed=len(executed),
                    finish_reason=response.finish_reason.value,
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
                return ToolLoopOutput(
                    answer        = response.content,
                    executed      = tuple(executed),
                    model         = response.model,
                    finish_reason = response.finish_reason,
                    tokens_used   = total_tokens,
                    iterations    = iteration,
                    truncated     = False,
                )

            # Execute each tool call against the domain executor.
            assistant_turn, tool_turns, new_records = await self._dispatch(
                tool_calls=response.tool_calls, tenant=tenant,
                assistant_content=response.content,
            )
            history.append(assistant_turn)
            history.extend(tool_turns)
            executed.extend(new_records)

        # Max iterations reached — force a final no-tools turn so the
        # model summarises whatever it has rather than looping forever.
        logger.warning(
            "mcp.tool_loop_truncated",
            tenant_id=tenant.tenant_id,
            iterations=iteration,
            executed=len(executed),
        )
        final = await self._call_provider(
            history, input_, tenant, use_tools=False,
        )
        total_tokens += final.usage.total_tokens
        return ToolLoopOutput(
            answer        = final.content or "I wasn't able to complete that request.",
            executed      = tuple(executed),
            model         = final.model or last_model,
            finish_reason = final.finish_reason or last_finish,
            tokens_used   = total_tokens,
            iterations    = iteration,
            truncated     = True,
        )

    # ── internal ──────────────────────────────────────────────────

    async def _call_provider(
        self,
        history:     List[LLMMessage],
        input_:      ToolLoopInput,
        tenant:      TenantContext,
        *,
        use_tools:   bool,
    ) -> LLMResponse:
        request = LLMRequest(
            messages    = tuple(history),
            model       = input_.model,
            max_tokens  = input_.max_tokens,
            temperature = input_.temperature,
            tools       = input_.tools if use_tools and input_.tools else None,
            tool_choice = (input_.tool_choice or "auto") if use_tools and input_.tools else None,
            stream      = False,
        )
        return await self._provider.complete(request, tenant=tenant)

    async def _dispatch(
        self,
        *,
        tool_calls:        Tuple[LLMToolCall, ...],
        tenant:            TenantContext,
        assistant_content: str,
    ) -> Tuple[LLMMessage, List[LLMMessage], List[ExecutedToolCall]]:
        """Execute each ``LLMToolCall`` and build the next turn's
        history (assistant turn carrying tool_calls + one role=tool
        message per result). Tool lookups go through the catalogue
        port so unknown tools fail fast with a structured result the
        LLM can read."""
        assistant_turn = LLMMessage(
            role        = MessageRole.ASSISTANT,
            content     = assistant_content or "",
            tool_calls  = tool_calls,
        )
        tool_turns: List[LLMMessage] = []
        records:    List[ExecutedToolCall] = []

        for tc in tool_calls:
            args = tc.args_or_empty
            tool = await self._catalogue.get(ToolName(tc.name))
            if tool is None:
                error_text = f"unknown tool: {tc.name}"
                tool_turns.append(LLMMessage(
                    role         = MessageRole.TOOL,
                    tool_call_id = tc.id,
                    name         = tc.name,
                    content      = error_text,
                ))
                records.append(ExecutedToolCall(
                    name=tc.name, arguments=args, result=error_text, is_error=True,
                ))
                continue

            try:
                result = await self._executor.execute(
                    tool=tool, arguments=args, tenant=tenant,
                )
            except Exception as e:  # noqa: BLE001
                error_text = f"tool failed: {type(e).__name__}: {e}"
                tool_turns.append(LLMMessage(
                    role         = MessageRole.TOOL,
                    tool_call_id = tc.id,
                    name         = tc.name,
                    content      = error_text,
                ))
                records.append(ExecutedToolCall(
                    name=tc.name, arguments=args, result=error_text, is_error=True,
                ))
                continue

            text = "\n".join(
                getattr(b, "text", "") for b in result.content
            ) or ""
            tool_turns.append(LLMMessage(
                role         = MessageRole.TOOL,
                tool_call_id = tc.id,
                name         = tc.name,
                content      = text,
            ))
            records.append(ExecutedToolCall(
                name=tc.name, arguments=args, result=text,
                is_error=result.is_error,
            ))

        return assistant_turn, tool_turns, records
