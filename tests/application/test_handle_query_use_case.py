"""HandleQueryUseCase — orchestration characterization tests.

HandleQueryUseCase owns the bot-query pipeline: policy guard ->
assemble context -> per-bot tools -> LLM+tool loop -> response mapping.
The three heavy components (assembler, tool registry, LLM router) are
injected, so we drive the orchestration with fakes and assert the
ordering, the argument threading, the chunk->source mapping, and the
policy denial — the invariants that must not drift when the legacy
MessageHandler is finally deleted.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.application.chat import HandleQueryInput, HandleQueryUseCase
from src.application.chat.handle_query import _chunks_to_sources
from src.domain.shared.tenant import TenantContext


class _Assembled:
    def __init__(self, messages: list[dict[str, Any]], chunks: list[Any]) -> None:
        self.messages = messages
        self.retrieved_chunks = chunks
        self.bot_config: dict[str, Any] = {}
        self.system_prompt = ""
        self.context_tokens = 0


class _Chunk:
    def __init__(self, content: str, score: float, kb_id: str = "kb1", kb_name: str = "KB One") -> None:
        self.content = content
        self.score = score
        self.kb_id = kb_id
        self.kb_name = kb_name
        self.document_id = "doc1"
        self.document_title = "Doc One"
        self.chunk_id = "c1"
        self.metadata = {"kb_name": kb_name}


class _ToolCall:
    def __init__(self, name: str) -> None:
        self._name = name

    def model_dump(self) -> dict[str, Any]:
        return {"tool_name": self._name}


class _Result:
    def __init__(self) -> None:
        self.answer = "the answer"
        self.tool_calls = [_ToolCall("rag_query")]
        self.tokens_used = 123
        self.model = "gemini/gemini-2.5-flash"
        self.finish_reason = "stop"


class _FakeAssembler:
    def __init__(self, assembled: _Assembled) -> None:
        self._a = assembled
        self.calls: list[dict[str, Any]] = []

    async def assemble(self, *, query, session, tenant_context, bot_id, custom_prompt=None):
        self.calls.append({"query": query, "bot_id": bot_id, "custom_prompt": custom_prompt})
        return self._a


class _FakeTools:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def get_tools_for_bot(self, *, bot_id, tenant_context, enabled_tools):
        self.calls.append({"bot_id": bot_id, "enabled_tools": enabled_tools})
        return []


class _FakeLLM:
    def __init__(self, result: _Result) -> None:
        self._r = result
        self.calls: list[dict[str, Any]] = []

    async def execute(self, *, messages, tools, tenant_context, stream, model):
        self.calls.append({"messages": messages, "stream": stream, "model": model})
        return self._r


def _tenant(tenant_id: str = "t-1") -> TenantContext:
    return TenantContext(
        tenant_id=tenant_id,
        user_id="u-1",
        user_email="a@shielva.ai",
        role="tenant_admin",
        permissions=(),
    )


@pytest.mark.asyncio
async def test_happy_path_orchestrates_and_maps_sources() -> None:
    assembler = _FakeAssembler(
        _Assembled(messages=[{"role": "user", "content": "hi"}], chunks=[_Chunk("some context", 0.912345)])
    )
    tools = _FakeTools()
    llm = _FakeLLM(_Result())
    uc = HandleQueryUseCase(context_assembler=assembler, tool_registry=tools, llm_router=llm)

    out = await uc.execute(
        input_=HandleQueryInput(query="hello", bot_id="bot-9", custom_prompt="be brief", model="gemini-2.5-pro"),
        tenant=_tenant(),
    )

    # ordering + argument threading
    assert assembler.calls[0]["bot_id"] == "bot-9"
    assert assembler.calls[0]["custom_prompt"] == "be brief"
    assert tools.calls[0]["bot_id"] == "bot-9"
    assert llm.calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert llm.calls[0]["model"] == "gemini-2.5-pro"

    # response shape
    assert out.answer == "the answer"
    assert out.model == "gemini/gemini-2.5-flash"
    assert out.tokens_used == 123
    assert out.tool_calls == [{"tool_name": "rag_query"}]

    # chunk -> source mapping (score rounded to 4dp, content present)
    assert len(out.sources) == 1
    src = out.sources[0]
    assert src["kb_id"] == "kb1"
    assert src["kb_name"] == "KB One"
    assert src["score"] == 0.9123
    assert src["content"] == "some context"


@pytest.mark.asyncio
async def test_no_tenant_id_is_denied() -> None:
    uc = HandleQueryUseCase(
        context_assembler=_FakeAssembler(_Assembled([], [])),
        tool_registry=_FakeTools(),
        llm_router=_FakeLLM(_Result()),
    )
    with pytest.raises(PermissionError):
        await uc.execute(input_=HandleQueryInput(query="x", bot_id="b"), tenant=_tenant(tenant_id=""))


@pytest.mark.asyncio
async def test_source_mapping_never_breaks_response() -> None:
    # A malformed chunk (content is None, no metadata) must not raise.
    bad = _Chunk("", 0.5)
    bad.content = None  # type: ignore[assignment]
    assembler = _FakeAssembler(_Assembled(messages=[], chunks=[bad]))
    uc = HandleQueryUseCase(context_assembler=assembler, tool_registry=_FakeTools(), llm_router=_FakeLLM(_Result()))
    out = await uc.execute(input_=HandleQueryInput(query="x", bot_id="b"), tenant=_tenant())
    assert out.answer == "the answer"  # response still produced


class _BoomLLM:
    async def execute(self, *, messages, tools, tenant_context, stream, model):
        raise ValueError("provider exploded")


@pytest.mark.asyncio
async def test_llm_failure_propagates() -> None:
    # A non-permission failure from the provider must propagate (and be logged),
    # not be swallowed — exercises the use case's generic-exception branch.
    uc = HandleQueryUseCase(
        context_assembler=_FakeAssembler(_Assembled(messages=[], chunks=[])),
        tool_registry=_FakeTools(),
        llm_router=_BoomLLM(),
    )
    with pytest.raises(ValueError, match="provider exploded"):
        await uc.execute(input_=HandleQueryInput(query="x", bot_id="b"), tenant=_tenant())


def test_chunks_to_sources_maps_and_trims() -> None:
    out = _chunks_to_sources([_Chunk("x" * 500, 0.987654)])
    assert len(out) == 1
    assert out[0]["content"] == "x" * 200  # trimmed to 200 chars for wire size
    assert out[0]["score"] == 0.9877  # rounded to 4dp


def test_chunks_to_sources_skips_bad_chunk_without_raising() -> None:
    good = _Chunk("ok", 0.5)
    bad = _Chunk("bad", 0.5)
    bad.score = "not-a-number"  # type: ignore[assignment]  # round() raises -> skipped
    out = _chunks_to_sources([bad, good])
    assert [s["content"] for s in out] == ["ok"]  # bad dropped, good kept


def test_chunks_to_sources_handles_none() -> None:
    assert _chunks_to_sources(None) == []
