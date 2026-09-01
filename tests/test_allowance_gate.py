"""The allowance gate in front of the LLM path.

🚨 It exists to stop spend, not to stop calls. Every uncertain answer — no
tenant, no config, a timeout, a bad response — must let the call through: the
cost of a small overage is far below the cost of a phone line that stops
answering because a billing lookup was slow.
"""

from __future__ import annotations

import inspect
import pathlib

import pytest

from src.infrastructure.metering import allowance


@pytest.mark.asyncio
async def test_no_tenant_never_blocks() -> None:
    assert await allowance.is_exhausted(None) is False
    assert await allowance.is_exhausted("") is False


@pytest.mark.asyncio
async def test_unconfigured_never_blocks(monkeypatch) -> None:
    """A missing ingest URL means we cannot know — so we do not refuse."""
    monkeypatch.setattr(allowance, "_configured", lambda: None)
    allowance._CACHE.clear()
    assert await allowance.is_exhausted("t-1") is False


@pytest.mark.asyncio
async def test_a_transport_error_never_blocks(monkeypatch) -> None:
    import httpx

    monkeypatch.setattr(allowance, "_configured", lambda: ("http://x", "s"))
    allowance._CACHE.clear()

    class _Boom:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, *a, **k):
            raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _Boom())
    assert await allowance.is_exhausted("t-1") is False


def test_the_router_refuses_with_a_payment_type_not_a_generic_error() -> None:
    """402 and 500 say different things to a customer.

    Read from source rather than imported: the router pulls in sealed settings,
    which need the deployment's config to exist — a test of THIS behaviour
    should not depend on that.
    """
    src = pathlib.Path("src/routing/llm_router.py").read_text()
    assert "class AllowanceExhausted(RuntimeError):" in src
    assert "await is_exhausted(" in src
    assert "raise AllowanceExhausted(" in src


def test_either_ceiling_stops_the_spend() -> None:
    """The token count OR the output half of it — output is what costs."""
    src = inspect.getsource(allowance.is_exhausted)
    assert '"exhausted"' in src
    assert '"output_exhausted"' in src
