"""Session aggregate — pure domain logic, no framework imports.

Tests prove:
    * State machine transitions (INITIALIZING → READY → CLOSING).
    * ``mark_ready`` is idempotent.
    * ``mark_ready`` from CLOSING raises SessionStateError.
    * ``allows_method`` enforces the spec's pre-initialize gate
      (only ``initialize`` and ``ping`` allowed when state is
      INITIALIZING).
"""
from __future__ import annotations

import pytest

from src.domain.chat.entities import Session
from src.domain.chat.errors import SessionStateError
from src.domain.chat.value_objects import (
    ClientInfo, ProtocolVersion, SessionId, SessionState,
)
from src.domain.shared.tenant import TenantContext


def _make_session() -> Session:
    return Session(
        id               = SessionId("test-session"),
        tenant           = TenantContext(
            tenant_id="t1", user_id="u1", user_email="u@example.com",
        ),
        protocol_version = ProtocolVersion("2024-11-05"),
        state            = SessionState.INITIALIZING,
        client_info      = ClientInfo(name="test-client", version="0.1"),
    )


class TestSessionStateMachine:
    def test_starts_initializing(self) -> None:
        s = _make_session()
        assert s.state is SessionState.INITIALIZING

    def test_mark_ready_transitions(self) -> None:
        s = _make_session()
        s.mark_ready()
        assert s.state is SessionState.READY

    def test_mark_ready_idempotent(self) -> None:
        s = _make_session()
        s.mark_ready()
        s.mark_ready()  # second call must NOT raise
        assert s.state is SessionState.READY

    def test_mark_ready_after_close_raises(self) -> None:
        s = _make_session()
        s.mark_closing()
        with pytest.raises(SessionStateError):
            s.mark_ready()

    def test_mark_closing_transitions(self) -> None:
        s = _make_session()
        s.mark_ready()
        s.mark_closing()
        assert s.state is SessionState.CLOSING


class TestAllowsMethod:
    @pytest.mark.parametrize("method", ["initialize", "ping"])
    def test_initialize_and_ping_allowed_pre_ready(self, method: str) -> None:
        s = _make_session()
        assert s.allows_method(method) is True

    @pytest.mark.parametrize("method", [
        "tools/list", "tools/call", "resources/list",
        "prompts/list", "logging/setLevel",
    ])
    def test_other_methods_blocked_pre_ready(self, method: str) -> None:
        s = _make_session()
        assert s.allows_method(method) is False

    def test_all_methods_allowed_after_ready(self) -> None:
        s = _make_session()
        s.mark_ready()
        for m in ("tools/list", "tools/call", "resources/list",
                  "prompts/list", "logging/setLevel"):
            assert s.allows_method(m) is True

    def test_no_methods_allowed_during_close(self) -> None:
        s = _make_session()
        s.mark_ready()
        s.mark_closing()
        assert s.allows_method("ping") is False
        assert s.allows_method("tools/list") is False


class TestActivityTracking:
    def test_touch_updates_last_activity(self) -> None:
        s = _make_session()
        before = s.last_activity_at
        s.touch(now=before + 100.0)
        assert s.last_activity_at == before + 100.0
