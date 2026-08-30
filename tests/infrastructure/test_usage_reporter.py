"""Usage metering must never break a completion.

Every test here asserts the same property from a different angle: whatever goes
wrong with billing telemetry, the customer still gets their answer. A lost
usage row is a reconciliation problem; a raised exception on the completion
path is a broken product.
"""

from __future__ import annotations

import asyncio

import pytest

from src.infrastructure.metering import usage_reporter
from src.infrastructure.metering.usage_reporter import report_llm_usage


@pytest.fixture(autouse=True)
def _quiet_config(monkeypatch: pytest.MonkeyPatch):
    """Default to unconfigured, so tests opt in to the network path."""
    monkeypatch.setattr(usage_reporter, "_configured", lambda: False)


def test_unconfigured_is_a_silent_no_op() -> None:
    """An unset ingest URL disables metering rather than failing."""
    report_llm_usage(tenant_id="t", total_tokens=100)  # must not raise


def test_missing_tenant_is_skipped() -> None:
    report_llm_usage(tenant_id=None, total_tokens=100)


def test_zero_tokens_is_not_reported() -> None:
    report_llm_usage(tenant_id="t", total_tokens=0)


def test_negative_tokens_is_not_reported() -> None:
    report_llm_usage(tenant_id="t", total_tokens=-5)


def test_no_running_loop_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Called from sync context with no loop — degrade, don't explode."""
    monkeypatch.setattr(usage_reporter, "_configured", lambda: True)
    report_llm_usage(tenant_id="t", total_tokens=10)


@pytest.mark.asyncio
async def test_transport_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ingest endpoint being down must not surface to the caller."""
    monkeypatch.setattr(usage_reporter, "_configured", lambda: True)

    async def _boom(payload):
        raise ConnectionError("ingest is down")

    monkeypatch.setattr(usage_reporter, "_post", _boom)
    report_llm_usage(tenant_id="t", total_tokens=10)
    # Let the scheduled task run; the exception must die inside it.
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_in_flight_task_is_strongly_referenced(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare create_task can be GC'd mid-flight and the write vanishes."""
    monkeypatch.setattr(usage_reporter, "_configured", lambda: True)
    started = asyncio.Event()

    async def _slow(payload):
        started.set()
        await asyncio.sleep(0.05)

    monkeypatch.setattr(usage_reporter, "_post", _slow)
    report_llm_usage(tenant_id="t", total_tokens=10)
    await started.wait()
    assert usage_reporter._IN_FLIGHT, "in-flight report was not retained"
    await asyncio.sleep(0.1)
    assert not usage_reporter._IN_FLIGHT, "completed report was not released"


@pytest.mark.asyncio
async def test_payload_shape_is_what_ingest_expects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usage_reporter, "_configured", lambda: True)
    seen: list[dict] = []

    async def _capture(payload):
        seen.append(payload)

    monkeypatch.setattr(usage_reporter, "_post", _capture)
    report_llm_usage(tenant_id="acme", total_tokens=1234, model_ref="m", request_id="req-9")
    await asyncio.sleep(0.05)

    assert len(seen) == 1
    ev = seen[0]["events"][0]
    assert ev["tenant_id"] == "acme"
    assert ev["kind"] == "llm"
    assert ev["quantity"] == 1234
    # A retried report must be absorbed rather than double-metered.
    assert ev["dedupe_key"] == "llm:req-9"
