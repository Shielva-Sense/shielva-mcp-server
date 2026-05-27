"""CompleteWithToolLoopUseCase — tests with fake LLM + tools.

Proves:
    * Loop terminates on a no-tool-calls response (happy path).
    * Loop executes tool calls, appends turn history, re-prompts.
    * Loop respects max_iterations and surfaces truncated=True.
    * Unknown tool name produces an is_error tool turn (not a crash).
    * Tool execution exception produces an is_error tool turn.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.application.llm import (
    CompleteWithToolLoopUseCase, ToolLoopInput,
)
from src.domain.llm.repositories import LLMProvider
from src.domain.llm.value_objects import (
    FinishReason, LLMMessage, LLMRequest, LLMResponse, LLMToolCall, LLMUsage,
    MessageRole, ModelId,
)
from src.domain.shared.tenant import TenantContext
from src.domain.tools.entities import Tool
from src.domain.tools.repositories import ToolCatalogue, ToolExecutor
from src.domain.tools.value_objects import (
    ToolName, ToolResult, ToolSchema,
)


# ── Fakes ───────────────────────────────────────────────────────────

class _ScriptedProvider(LLMProvider):
    """Returns the next response from a queued script. Useful for
    exercising loop turns deterministically."""

    def __init__(self, script: List[LLMResponse]) -> None:
        self._script = list(script)
        self.calls: List[LLMRequest] = []

    async def complete(self, request: LLMRequest, *, tenant: TenantContext
                       ) -> LLMResponse:
        self.calls.append(request)
        if not self._script:
            # Default — force a clean termination if the loop asked
            # for "one more turn" beyond what the test scripted.
            return LLMResponse(
                content       = "done",
                finish_reason = FinishReason.STOP,
                model         = ModelId("test"),
            )
        return self._script.pop(0)


class _FakeCatalogue(ToolCatalogue):
    def __init__(self, tools: List[Tool]) -> None:
        self._tools = {str(t.name): t for t in tools}

    async def list_for(self, tenant: TenantContext) -> List[Tool]:
        return list(self._tools.values())

    async def get(self, name: ToolName) -> Optional[Tool]:
        return self._tools.get(str(name))


class _FakeExecutor(ToolExecutor):
    def __init__(self, behaviour: Dict[str, Any] | None = None) -> None:
        self._behaviour = behaviour or {}
        self.executions: List[tuple[str, Dict[str, Any]]] = []

    async def execute(self, *, tool, arguments, tenant, context=None
                      ) -> ToolResult:
        self.executions.append((str(tool.name), dict(arguments)))
        recipe = self._behaviour.get(str(tool.name))
        if recipe is None:
            return ToolResult.text(f"ok:{tool.name}")
        if isinstance(recipe, Exception):
            raise recipe
        return recipe


# ── Helpers ─────────────────────────────────────────────────────────

def _tool(name: str) -> Tool:
    return Tool(
        name                 = ToolName(name),
        description          = name,
        input_schema         = ToolSchema(json_schema={"type": "object"}),
        required_permissions = tuple(),
    )


def _tenant() -> TenantContext:
    return TenantContext(tenant_id="t1", user_id="u", user_email="u@x")


def _user_msg(text: str) -> LLMMessage:
    return LLMMessage(role=MessageRole.USER, content=text)


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content       = text,
        finish_reason = FinishReason.STOP,
        model         = ModelId("test"),
        usage         = LLMUsage(total_tokens=10),
    )


def _tool_call_response(tc_id: str, name: str, args: str = "{}") -> LLMResponse:
    return LLMResponse(
        content       = "",
        tool_calls    = (LLMToolCall(id=tc_id, name=name, arguments=args),),
        finish_reason = FinishReason.TOOL_CALLS,
        model         = ModelId("test"),
        usage         = LLMUsage(total_tokens=20),
    )


def _use_case(*, script: List[LLMResponse], tools: List[Tool] | None = None,
              executor_behaviour: Dict[str, Any] | None = None,
              ) -> tuple[CompleteWithToolLoopUseCase, _ScriptedProvider, _FakeExecutor]:
    provider = _ScriptedProvider(script=script)
    catalogue = _FakeCatalogue(tools or [_tool("dummy")])
    executor  = _FakeExecutor(behaviour=executor_behaviour)
    uc = CompleteWithToolLoopUseCase(
        provider=provider, catalogue=catalogue, executor=executor,
    )
    return uc, provider, executor


# ── Tests ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_loop_returns_text_immediately_when_no_tool_calls():
    uc, provider, executor = _use_case(script=[_text_response("hello")])
    out = await uc.execute(
        input_=ToolLoopInput(
            messages=(_user_msg("hi"),),
            tools=tuple(),
        ),
        tenant=_tenant(),
    )
    assert out.answer == "hello"
    assert out.iterations == 1
    assert out.truncated is False
    assert out.executed == ()
    assert len(provider.calls) == 1
    assert len(executor.executions) == 0


@pytest.mark.asyncio
async def test_loop_executes_tool_then_summarises():
    """Turn 1: model emits tool_call. Loop runs the tool, re-prompts.
    Turn 2: model returns text — loop terminates."""
    uc, provider, executor = _use_case(
        script=[
            _tool_call_response("tc-1", "lookup", '{"q":"shielva"}'),
            _text_response("Found: ok:lookup"),
        ],
        tools=[_tool("lookup")],
    )
    out = await uc.execute(
        input_=ToolLoopInput(
            messages=(_user_msg("look up shielva"),),
            tools=({"type":"function","function":{"name":"lookup","description":"x","parameters":{}}},),
        ),
        tenant=_tenant(),
    )
    assert "Found: ok:lookup" in out.answer
    assert out.iterations == 2
    assert out.truncated is False
    assert len(out.executed) == 1
    assert out.executed[0].name == "lookup"
    assert out.executed[0].arguments == {"q": "shielva"}
    assert out.executed[0].is_error is False
    # Provider call 2 must see the assistant tool_calls turn + a
    # role=tool message — confirm message growth.
    assert len(provider.calls[1].messages) == len(provider.calls[0].messages) + 2


@pytest.mark.asyncio
async def test_loop_truncates_after_max_iterations():
    """Model keeps emitting tool_calls; loop forces a no-tools turn
    after max_iterations and reports truncated=True.

    Script: exactly ``max_iterations`` tool-using turns + 1 final
    text turn for the forced no-tools summary call.
    """
    script = [
        _tool_call_response(f"tc-{i}", "looping", "{}") for i in range(3)
    ] + [_text_response("giving up")]
    uc, provider, _ = _use_case(
        script=script,
        tools=[_tool("looping")],
    )
    out = await uc.execute(
        input_=ToolLoopInput(
            messages=(_user_msg("go"),),
            tools=({"type":"function","function":{"name":"looping","description":"x","parameters":{}}},),
            max_iterations=3,
        ),
        tenant=_tenant(),
    )
    assert out.truncated is True
    assert out.iterations == 3
    assert out.answer == "giving up"
    # 3 tool-using turns + 1 final no-tools turn = 4 provider calls.
    assert len(provider.calls) == 4
    # The final call must NOT have tools attached (we forced a
    # summary).
    assert provider.calls[-1].tools is None


@pytest.mark.asyncio
async def test_unknown_tool_call_returns_error_tool_message_not_crash():
    """When the LLM hallucinates a tool name the loop must NOT
    crash; it must inject an error tool message so the LLM can see
    what happened and choose a different tool."""
    uc, provider, executor = _use_case(
        script=[
            _tool_call_response("tc-1", "does_not_exist", "{}"),
            _text_response("recovered"),
        ],
        tools=[_tool("real")],   # different name on purpose
    )
    out = await uc.execute(
        input_=ToolLoopInput(
            messages=(_user_msg("x"),),
            tools=({"type":"function","function":{"name":"real","description":"x","parameters":{}}},),
        ),
        tenant=_tenant(),
    )
    assert out.answer == "recovered"
    assert len(out.executed) == 1
    assert out.executed[0].is_error is True
    assert "unknown tool" in out.executed[0].result
    # No real execution happened.
    assert len(executor.executions) == 0


@pytest.mark.asyncio
async def test_tool_execution_failure_surfaces_as_error_record():
    uc, provider, executor = _use_case(
        script=[
            _tool_call_response("tc-1", "boom", "{}"),
            _text_response("recovered after failure"),
        ],
        tools=[_tool("boom")],
        executor_behaviour={"boom": RuntimeError("crash")},
    )
    out = await uc.execute(
        input_=ToolLoopInput(
            messages=(_user_msg("x"),),
            tools=({"type":"function","function":{"name":"boom","description":"x","parameters":{}}},),
        ),
        tenant=_tenant(),
    )
    assert out.answer == "recovered after failure"
    assert out.executed[0].is_error is True
    assert "RuntimeError" in out.executed[0].result
