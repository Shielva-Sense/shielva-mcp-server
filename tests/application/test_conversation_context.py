"""Conversation-memory hydration for ``/mcp/v1/query``.

``SessionContext`` was built empty on every request, so the context assembler —
which already replays ``session.messages`` — always replayed nothing and every
RAG turn was a cold start. ``_conversation_messages`` is the mapping that closes
that gap, translating the caller-supplied transcript into protocol messages.

These are the invariants that must not drift:
  * a rolling summary leads, so long chats stay coherent without full replay
  * caller order is preserved (a reordered transcript changes the answer)
  * `tool` roles are dropped — replaying them without their call ids produces
    orphaned tool messages at the provider
  * one malformed turn is skipped, never fatal (same contract as
    ``_chunks_to_sources``): a bad row in memory must not fail the user's query
"""

from __future__ import annotations

from src.application.chat.handle_query import _conversation_messages
from src.protocol.models import MessageRole


def test_no_context_yields_no_messages():
    assert _conversation_messages(None) == []
    assert _conversation_messages({}) == []


def test_messages_are_mapped_in_caller_order():
    out = _conversation_messages(
        {
            "messages": [
                {"role": "user", "content": "what is kubernetes"},
                {"role": "assistant", "content": "a container orchestrator"},
                {"role": "user", "content": "and pods?"},
            ]
        }
    )
    assert [m.role for m in out] == [MessageRole.USER, MessageRole.ASSISTANT, MessageRole.USER]
    assert [m.content for m in out] == ["what is kubernetes", "a container orchestrator", "and pods?"]


def test_summary_is_prepended_as_a_system_message():
    out = _conversation_messages(
        {"summary": "user is deploying to k3s", "messages": [{"role": "user", "content": "next step?"}]}
    )
    assert out[0].role is MessageRole.SYSTEM
    assert "user is deploying to k3s" in out[0].content
    assert out[1].content == "next step?"


def test_summary_alone_still_produces_context():
    out = _conversation_messages({"summary": "prior chat about billing"})
    assert len(out) == 1
    assert out[0].role is MessageRole.SYSTEM


def test_blank_summary_adds_nothing():
    out = _conversation_messages({"summary": "   ", "messages": [{"role": "user", "content": "hi"}]})
    assert len(out) == 1
    assert out[0].role is MessageRole.USER


def test_tool_role_is_dropped():
    out = _conversation_messages(
        {
            "messages": [
                {"role": "user", "content": "search"},
                {"role": "tool", "content": '{"call_id": "abc"}'},
                {"role": "assistant", "content": "done"},
            ]
        }
    )
    assert [m.role for m in out] == [MessageRole.USER, MessageRole.ASSISTANT]


def test_system_role_is_allowed():
    out = _conversation_messages({"messages": [{"role": "system", "content": "be terse"}]})
    assert out[0].role is MessageRole.SYSTEM


def test_unknown_role_is_skipped():
    out = _conversation_messages({"messages": [{"role": "wizard", "content": "x"}, {"role": "user", "content": "y"}]})
    assert [m.content for m in out] == ["y"]


def test_empty_and_whitespace_content_are_skipped():
    out = _conversation_messages(
        {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "user", "content": "   "},
                {"role": "user", "content": "real"},
            ]
        }
    )
    assert [m.content for m in out] == ["real"]


def test_malformed_entry_is_skipped_not_fatal():
    """A row that isn't a dict must not take down the whole query."""
    out = _conversation_messages({"messages": ["not-a-dict", {"role": "user", "content": "survives"}]})
    assert [m.content for m in out] == ["survives"]


def test_non_list_messages_is_tolerated():
    assert _conversation_messages({"messages": None}) == []


def test_content_is_stringified_and_trimmed():
    out = _conversation_messages({"messages": [{"role": "user", "content": "  padded  "}]})
    assert out[0].content == "padded"
