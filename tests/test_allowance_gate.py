"""The allowance gate in front of the LLM path.

🚨 It exists to stop spend, not to stop calls. Every uncertain answer — no
tenant, no config, a timeout, a bad response — must let the call through: the
cost of a small overage is far below the cost of a phone line that stops
answering because a billing lookup was slow.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re
import textwrap

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


# ── The gate must sit on the path that actually runs ────────────────────────


def test_the_live_adapter_gates_before_it_calls_the_vendor() -> None:
    """🚨 The gate lived ONLY in llm_router, and the LiteLLM adapter calls
    `acompletion` directly — it uses the router to resolve a model, not to make
    the call. Metering was moved into the adapter when LLM usage stopped being
    recorded, but the CEILING was left behind, so a workspace could spend past
    both the included tokens and the output cap and only be billed after.

    Pinned by position, not just presence: a check that runs after the vendor
    call has already spent the money it exists to prevent.
    """
    from src.infrastructure.llm import litellm_provider as lp

    for fn in (lp.LiteLLMProviderAdapter.complete, lp.LiteLLMProviderAdapter.stream):
        src = inspect.getsource(fn)
        assert "is_exhausted(" in src, f"{fn.__name__} does not check the allowance"
        # 🚨 Compare against CODE, not the docstring. stream() documents itself
        # as "via litellm.acompletion(stream=True)", and matching that made the
        # gate look mis-ordered when it was placed correctly.
        body = ast.get_source_segment(src, ast.parse(textwrap.dedent(src)).body[0]) or src
        stripped = re.sub(r'"""(?:.|\n)*?"""', "", body, count=1)
        assert stripped.index("is_exhausted(") < stripped.index("acompletion("), (
            f"{fn.__name__} checks the allowance after calling the vendor"
        )


def test_the_gate_stops_on_either_ceiling() -> None:
    """Included tokens and the output cap are separate limits. Output costs
    several times input at every vendor, so a chatty workload can blow the cap
    long before the token count runs out."""
    src = pathlib.Path(allowance.__file__).read_text()
    assert '"exhausted"' in src
    assert '"output_exhausted"' in src
