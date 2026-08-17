"""Token streaming on the query path, and the grounding it must not lose.

Measured on a live phone call: the batched path is 1,754ms of a 2,261ms turn —
78% — because the caller hears nothing until the whole answer exists. Streaming
makes that time-to-first-token.

Two earlier attempts at streaming in this service were reverted, both for the
same shape of bug: `routing.llm_router._execute_streaming` silently dropped the
tool-calling loop, and `text_only=True` silently dropped RAG. The failure mode
is a FAST UNGROUNDED ANSWER, which is worse for a caller than a slow correct
one — so the property pinned hardest here is that retrieval still happens.
"""

from __future__ import annotations

import pytest

from src.application.chat import HandleQueryInput, HandleQueryUseCase


class _Ctx:
    def __init__(self, chunks):
        self.messages = [{"role": "user", "content": "what are your timings"}]
        self.retrieved_chunks = chunks


class _Assembler:
    """Records that it ran — assembly is where RAG and fencing live."""

    def __init__(self, chunks=("chunk-a", "chunk-b")):
        self.called = False
        self._chunks = list(chunks)

    async def assemble(self, **kwargs):
        self.called = True
        self.kwargs = kwargs
        return _Ctx(self._chunks)


class _Tools:
    async def get_tools_for_bot(self, **kwargs):
        return []


class _Chunk:
    def __init__(self, delta):
        self.delta = delta


class _LLMService:
    def __init__(self, deltas=("Hello", " there")):
        self._deltas = list(deltas)

    async def stream(self, request, *, tenant):
        self.request = request
        for d in self._deltas:
            yield _Chunk(d)


class _Tenant:
    tenant_id = "Tenant-1"
    user_id = "u"
    user_email = "u@example.com"
    role = "Tenant Admin"
    permissions: list[str] = []


def _uc(assembler=None, llm_service=None):
    return HandleQueryUseCase(
        context_assembler=assembler or _Assembler(),
        tool_registry=_Tools(),
        llm_router=object(),
        llm_service=llm_service or _LLMService(),
    )


async def _drain(uc):
    return [ev async for ev in uc.execute_stream(
        input_=HandleQueryInput(query="what are your timings", bot_id="b"),
        tenant=_Tenant(),
    )]


@pytest.mark.asyncio
async def test_retrieval_still_runs_when_streaming():
    """The whole point. Assembly is where RAG happens; skipping it is how both
    previous attempts produced ungrounded answers."""
    asm = _Assembler()
    await _drain(_uc(assembler=asm))
    assert asm.called, "context assembly was skipped — the answer would be ungrounded"


@pytest.mark.asyncio
async def test_meta_arrives_before_any_token():
    """A consumer decides whether to speak at all from meta. After the first
    token that decision is already too late."""
    events = await _drain(_uc())
    kinds = [k for k, _ in events]
    assert kinds[0] == "meta"
    assert kinds.index("meta") < kinds.index("token")


@pytest.mark.asyncio
async def test_meta_reports_whether_the_answer_is_grounded():
    grounded = await _drain(_uc(assembler=_Assembler(["a"])))
    ungrounded = await _drain(_uc(assembler=_Assembler([])))
    assert grounded[0][1]["grounded"] is True
    assert ungrounded[0][1]["grounded"] is False


@pytest.mark.asyncio
async def test_tokens_arrive_individually_and_assemble_into_the_answer():
    """If deltas were aggregated before yielding, this would be one token and
    the latency win would be zero — the exact bug being fixed."""
    events = await _drain(_uc(llm_service=_LLMService(["Hel", "lo", " world"])))
    tokens = [v for k, v in events if k == "token"]
    assert len(tokens) == 3, "deltas were batched; there is no latency win"
    done = next(v for k, v in events if k == "done")
    assert done["answer"] == "Hello world"


@pytest.mark.asyncio
async def test_it_refuses_without_an_authenticated_principal():
    class _Anon(_Tenant):
        tenant_id = ""

    uc = _uc()
    with pytest.raises(PermissionError):
        async for _ in uc.execute_stream(
            input_=HandleQueryInput(query="q", bot_id="b"), tenant=_Anon()
        ):
            pass


@pytest.mark.asyncio
async def test_it_refuses_rather_than_answering_some_other_way():
    """Without the DDD service there is no token source. Falling back to the
    batched router would look like streaming and silently not be."""
    uc = HandleQueryUseCase(
        context_assembler=_Assembler(), tool_registry=_Tools(), llm_router=object()
    )
    with pytest.raises(RuntimeError, match="LLM service"):
        async for _ in uc.execute_stream(
            input_=HandleQueryInput(query="q", bot_id="b"), tenant=_Tenant()
        ):
            pass
