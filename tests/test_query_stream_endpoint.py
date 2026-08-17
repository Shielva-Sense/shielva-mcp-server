"""The SSE wire format of /mcp/v1/query/stream.

The use-case tests cover the decision to stream; these cover what actually
reaches a consumer, which is a different failure surface. A malformed frame or
a stream that ends without a terminal event does not raise anywhere — the
consumer simply waits, and on a phone call that is heard as a dropped line.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# The router imports `_deps`, which imports shielva_common — present in CI and
# in the image, not necessarily in a bare local checkout. Skipping beats an
# import error that looks like a broken test suite.
pytest.importorskip("shielva_common", reason="shielva_common not installed locally")

from src.infrastructure.entitlements import require_llm_entitlement
from src.interface.http._deps import get_domain_tenant
from src.interface.http.query_router import query_router


class _Tenant:
    tenant_id = "Tenant-1"
    user_id = "u"
    user_email = "u@example.com"
    role = "Tenant Admin"
    permissions: list[str] = []


class _StreamingUseCase:
    """Yields the (kind, payload) tuples the endpoint frames as SSE."""

    def __init__(self, events=None, raises: Exception | None = None):
        self._events = events or [
            ("meta", {"retrieved_chunks": 2, "grounded": True}),
            ("token", "Hello"),
            ("token", " there"),
            ("done", {"answer": "Hello there", "grounded": True}),
        ]
        self._raises = raises

    async def execute_stream(self, *, input_, tenant):
        for ev in self._events:
            yield ev
        if self._raises:
            raise self._raises

    async def execute(self, *, input_, tenant):  # batched path
        class _Out:
            answer = "batched answer"

        return _Out()


def _client(use_case, *, token_stream: bool, monkeypatch) -> TestClient:
    monkeypatch.setenv("MCP_QUERY_TOKEN_STREAM", "1" if token_stream else "")
    app = FastAPI()
    app.include_router(query_router)
    app.state.handle_query_use_case = use_case

    def _tenant() -> _Tenant:
        return _Tenant()

    def _no_entitlement_check() -> None:
        """Entitlement is enforced elsewhere; this test is about the wire."""

    app.dependency_overrides[get_domain_tenant] = _tenant
    app.dependency_overrides[require_llm_entitlement] = _no_entitlement_check
    return TestClient(app)


def _post(client: TestClient):
    return client.post("/mcp/v1/query/stream", json={"query": "timings", "bot_id": "b"})


def _frames(body: str) -> list[tuple[str, object]]:
    """Parse SSE back into (event, payload) — the consumer's view."""
    out = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        kind = payload = None
        for line in block.splitlines():
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
        if kind:
            out.append((kind, payload))
    return out


def test_token_streaming_emits_meta_then_tokens_then_done(monkeypatch):
    r = _post(_client(_StreamingUseCase(), token_stream=True, monkeypatch=monkeypatch))
    assert r.status_code == 200
    kinds = [k for k, _ in _frames(r.text)]
    assert kinds == ["meta", "token", "token", "done"]


def test_frames_are_parseable_sse(monkeypatch):
    """Each frame needs `event:`, `data:` and a terminating blank line. Without
    the blank line a consumer waits for a frame already sent."""
    r = _post(_client(_StreamingUseCase(), token_stream=True, monkeypatch=monkeypatch))
    assert "event: token\ndata: " in r.text
    assert r.text.endswith("\n\n")
    meta = _frames(r.text)[0][1]
    assert meta["grounded"] is True


def test_the_flag_off_keeps_the_old_single_event_behaviour(monkeypatch):
    """Default must not change. This is what every consumer runs on today."""
    r = _post(_client(_StreamingUseCase(), token_stream=False, monkeypatch=monkeypatch))
    assert r.status_code == 200
    assert r.text == "batched answer"
    assert "event: token" not in r.text


def test_a_failure_mid_stream_still_terminates(monkeypatch):
    """The consumer has already been told to expect tokens; dying silently
    strands it. A terminal `done` carrying the error lets it fall back."""
    uc = _StreamingUseCase(raises=RuntimeError("provider exploded"))
    r = _post(_client(uc, token_stream=True, monkeypatch=monkeypatch))
    frames = _frames(r.text)
    assert frames[-1][0] == "done"
    assert "error" in frames[-1][1]


@pytest.mark.parametrize("raw", ["1", "true", "YES", "on"])
def test_the_flag_accepts_the_usual_spellings(monkeypatch, raw):
    r = _post(_client(_StreamingUseCase(), token_stream=False, monkeypatch=monkeypatch))
    monkeypatch.setenv("MCP_QUERY_TOKEN_STREAM", raw)
    r = _post(_client(_StreamingUseCase(), token_stream=True, monkeypatch=monkeypatch))
    assert "event: token" in r.text
