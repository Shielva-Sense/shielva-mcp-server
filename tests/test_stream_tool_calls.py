"""A streamed completion may call tools: the words first, the whole calls at the end.

🚨 Why: a phone call speaks the model's first sentence while the rest is still
being written. That only works if the streaming endpoint takes the same tools as
``/llm/complete``; text-only streaming forced every tool-using caller back onto
the blocking path, where the caller hears nothing until the whole reply exists.

What these hold onto:

  * tool-call fragments (id and name once, arguments a few characters at a
    time) come out as ONE whole call, never partial JSON;
  * they arrive on the terminal chunk only, after the provider's stream ends,
    even when the provider sent ``finish_reason`` before the last fragment;
  * text said before the calls streams as it comes;
  * a request with no tools streams exactly as before (no tools sent).
"""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from src.domain.llm.value_objects import FinishReason, LLMMessage, LLMRequest, MessageRole
from src.domain.shared.tenant import TenantContext
from src.infrastructure.llm import litellm_provider
from src.infrastructure.llm.litellm_provider import LiteLLMProviderAdapter

TOOLS = ({"type": "function", "function": {"name": "find_times", "parameters": {"type": "object"}}},)


def _frag(index, *, id=None, name=None, args=None):  # noqa: A002 - the provider's own field name
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


def _part(text="", calls=None, finish=None):
    return NS(usage=None, choices=[NS(delta=NS(content=text, tool_calls=calls), finish_reason=finish)])


class _Router:
    default_model = "gemini/test"
    fallback_models: list[str] = []

    def _get_api_key(self, _model):
        return "k"


@pytest.fixture
def stream_of(monkeypatch):
    """Make `acompletion(stream=True)` replay `parts`, recording the kwargs it was called with."""
    seen: dict = {}

    def _install(parts):
        async def _gen():
            for p in parts:
                yield p

        async def _acompletion(**kw):
            seen.update(kw)
            return _gen()

        import litellm

        monkeypatch.setattr(litellm, "acompletion", _acompletion)

        async def _never(_tid):
            return False

        from src.infrastructure.metering import allowance

        monkeypatch.setattr(allowance, "is_exhausted", _never)

        class _Resolver:
            async def resolve(self, _tid):
                return None

        monkeypatch.setattr(litellm_provider, "get_tenant_llm_resolver", _Resolver)
        monkeypatch.setattr(litellm_provider, "report_llm_usage", lambda **_kw: None)
        return seen

    return _install


def _request(tools=TOOLS):
    return LLMRequest(
        messages=(LLMMessage(role=MessageRole.USER, content="any time tomorrow evening?"),),
        tools=tools,
        stream=True,
    )


TENANT = TenantContext(tenant_id="t-1", user_id="u", user_email="u@example.test", role="r")


async def _collect(parts, stream_of, request):
    seen = stream_of(parts)
    out = [c async for c in LiteLLMProviderAdapter(_Router()).stream(request, tenant=TENANT)]
    return out, seen


@pytest.mark.asyncio
async def test_the_words_stream_and_the_call_arrives_whole_at_the_end(stream_of):
    parts = [
        _part("One moment, "),
        _part("let me check."),
        _part(calls=[_frag(0, id="c1", name="find_times", args='{"when": "tom')]),
        # 🚨 Some providers say finish before the last fragment.
        _part(calls=[_frag(0, args='orrow evening"}')], finish="tool_calls"),
    ]
    out, seen = await _collect(parts, stream_of, _request())
    assert [c.delta for c in out[:-1]] == ["One moment, ", "let me check."]
    assert all(not c.tool_calls for c in out[:-1])
    last = out[-1]
    assert last.finish_reason == FinishReason.TOOL_CALLS
    assert len(last.tool_calls) == 1
    call = last.tool_calls[0]
    assert (call.id, call.name) == ("c1", "find_times")
    assert json.loads(call.arguments) == {"when": "tomorrow evening"}
    assert seen["tools"] == list(TOOLS)
    assert seen["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_two_calls_keep_their_order(stream_of):
    parts = [
        _part(calls=[_frag(1, id="b", name="book", args="{}"), _frag(0, id="a", name="find_times", args="{}")]),
        _part(finish="tool_calls"),
    ]
    out, _ = await _collect(parts, stream_of, _request())
    assert [c.name for c in out[-1].tool_calls] == ["find_times", "book"]


@pytest.mark.asyncio
async def test_a_plain_answer_has_no_calls_and_ends_with_stop(stream_of):
    out, _ = await _collect([_part("We open at nine."), _part(finish="stop")], stream_of, _request())
    assert out[-1].finish_reason == FinishReason.STOP
    assert out[-1].tool_calls == ()
    assert "".join(c.delta for c in out) == "We open at nine."


@pytest.mark.asyncio
async def test_no_tools_means_none_are_sent(stream_of):
    out, seen = await _collect([_part("Hello.")], stream_of, _request(tools=None))
    assert "tools" not in seen
    assert "tool_choice" not in seen
    assert "".join(c.delta for c in out) == "Hello."


def test_the_endpoint_takes_tools_and_sends_the_calls_on_its_last_frame():
    """The wire: tools reach the service, and the final frame carries the calls in `/llm/complete`'s shape."""
    import importlib

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.domain.llm.value_objects import LLMStreamChunk, LLMToolCall, ModelId
    from src.infrastructure.entitlements import require_llm_entitlement

    wire = importlib.import_module("src.interface.http.llm_router")
    from src.interface.http._deps import get_domain_tenant

    got: list[LLMRequest] = []

    class _Service:
        async def stream(self, request, *, tenant):
            got.append(request)
            yield LLMStreamChunk(delta="Let me check.", model=ModelId("m"))
            yield LLMStreamChunk(
                finish_reason=FinishReason.TOOL_CALLS,
                model=ModelId("m"),
                tool_calls=(LLMToolCall(id="c1", name="find_times", arguments='{"when": "tomorrow"}'),),
            )

    app = FastAPI()
    app.include_router(wire.llm_router)
    app.state.llm_app_service = _Service()
    app.dependency_overrides[get_domain_tenant] = lambda: TENANT
    app.dependency_overrides[require_llm_entitlement] = lambda: None
    body = {"messages": [{"role": "user", "content": "hi"}], "tools": list(TOOLS)}
    with TestClient(app) as client:
        resp = client.post(f"{wire._PREFIX}/llm/complete/stream", json=body)
    assert resp.status_code == 200
    frames = [line[6:] for line in resp.text.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    first, last = json.loads(frames[0]), json.loads(frames[-2])
    assert first["delta"] == "Let me check."
    assert last["finish_reason"] == "tool_calls"
    assert last["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "find_times", "arguments": '{"when": "tomorrow"}'}}
    ]
    assert got[0].tools == TOOLS
