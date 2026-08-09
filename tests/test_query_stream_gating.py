"""Streaming must actually fire, and must never cost a tool-enabled bot its tools.

The gate here is subtle and was nearly shipped wrong. Every tool registers with
``enabled_by_default=True`` and ``configure_bot_tools`` is never called, so
``get_tools_for_bot`` returns a NON-EMPTY list for every bot. A "stream only when
the bot has no tools" condition would therefore never be true and the whole
streaming path would be dead code.

So streaming is opt-in per REQUEST (``text_only``), and these tests pin both
halves: that opting in streams even though the bot has tools registered, and
that NOT opting in still runs the tool loop.
"""

from __future__ import annotations

import pytest

from src.application.chat import HandleQueryInput, HandleQueryUseCase
from src.domain.shared.tenant import TenantContext


class _Ctx:
    messages = [type("M", (), {"role": "user", "content": "hi"})()]
    retrieved_chunks = []


class _Assembler:
    async def assemble(self, **_kw):
        return _Ctx()


class _Registry:
    """Mirrors production: a bot always resolves a non-empty tool set."""

    def __init__(self, tools=None):
        self.tools = tools if tools is not None else ["rag_query", "current_time"]

    async def get_tools_for_bot(self, **_kw):
        return list(self.tools)


class _Router:
    def __init__(self):
        self.calls = []

    async def execute(self, **kw):
        self.calls.append(kw)
        return type("R", (), {"answer": "batched answer", "tool_calls": [], "tokens_used": 1, "model": "m"})()


class _Provider:
    def __init__(self):
        self.called = False

    async def stream(self, _req, *, tenant):  # noqa: ARG002
        self.called = True
        for tok in ("Strea", "med ", "answer."):
            yield type("C", (), {"delta": tok})()


def _uc(provider=None, tools=None):
    return HandleQueryUseCase(
        context_assembler=_Assembler(),
        tool_registry=_Registry(tools),
        llm_router=_Router(),
        llm_provider=provider,
    )


_TENANT = TenantContext(tenant_id="t1", user_id="u", user_email="u@x.com", role="member", permissions=[])


async def _run(uc, **over):
    kw = {"query": "q", "bot_id": "b", **over}
    return [e async for e in uc.execute_stream(input_=HandleQueryInput(**kw), tenant=_TENANT)]


@pytest.mark.asyncio
async def test_text_only_streams_even_though_the_bot_has_tools():
    """THE regression guard. Without this the gate is dead code in production."""
    provider = _Provider()
    events = await _run(_uc(provider), text_only=True)
    assert provider.called, "did not stream — the gate never fired"
    tokens = [e for e in events if e["event"] == "token"]
    assert len(tokens) > 1, "arrived as one event, so nothing was gained"
    assert "".join(t["data"]["text"] for t in tokens) == "Streamed answer."


@pytest.mark.asyncio
async def test_without_text_only_the_tool_loop_still_runs():
    """A tool-enabled bot must keep its tools — a fast empty answer is worse
    than a slow correct one."""
    provider = _Provider()
    uc = _uc(provider)
    events = await _run(uc)
    assert not provider.called
    assert uc._llm.calls, "batched path was not used"
    assert uc._llm.calls[0]["tools"] == ["rag_query", "current_time"]
    assert [e["event"] for e in events][-1] == "done"


@pytest.mark.asyncio
async def test_text_only_drops_tools_from_the_batched_fallback_too():
    """If the stream yields nothing we fall back — and must not silently
    reintroduce tools the caller asked to skip."""

    class _Empty(_Provider):
        async def stream(self, _req, *, tenant):  # noqa: ARG002
            self.called = True
            return
            yield  # pragma: no cover

    uc = _uc(_Empty())
    await _run(uc, text_only=True)
    assert uc._llm.calls[0]["tools"] == []


@pytest.mark.asyncio
async def test_meta_arrives_before_any_token():
    """Consumers gate auto-respond vs approval on meta BEFORE speaking."""
    events = await _run(_uc(_Provider()), text_only=True)
    assert events[0]["event"] == "meta"
    assert "confidence" in events[0]["data"]


@pytest.mark.asyncio
async def test_done_carries_the_assembled_text():
    events = await _run(_uc(_Provider()), text_only=True)
    done = events[-1]
    assert done["event"] == "done"
    assert done["data"]["text"] == "Streamed answer."


@pytest.mark.asyncio
async def test_no_streaming_provider_falls_back_cleanly():
    """Deployments wired without the provider must still answer."""
    events = await _run(_uc(provider=None), text_only=True)
    assert [e["event"] for e in events][-1] == "done"
    assert events[-1]["data"]["text"] == "batched answer"


@pytest.mark.asyncio
async def test_a_mid_stream_error_still_terminates_with_done():
    """A consumer mid-utterance needs a terminal event, not a dangling stream."""

    class _Boom(_Provider):
        async def stream(self, _req, *, tenant):  # noqa: ARG002
            self.called = True
            yield type("C", (), {"delta": "Partial "})()
            raise RuntimeError("provider died")

    events = await _run(_uc(_Boom()), text_only=True)
    assert events[-1]["event"] == "done"
    assert "Partial" in events[-1]["data"]["text"]
