"""A scanned page is read by the workspace's OWN provisioned model.

🚨 Why this exists instead of an OCR vendor. When a PDF has no text layer the
extractor has nothing to work with, and the row comes back saying so. The
provisioned model for every workspace on this platform is Gemini, which reads
images as first-class input — so the page can simply be RENDERED and sent. That
removes a vendor, a key, a per-page bill and a second pipeline, and the call is
metered exactly like every other LLM call because it IS one.

What the tests hold onto:

  * `parts` is additive. `content` stays a string for every existing caller —
    the tool loop, the token accountant, every prompt in the fleet read it as
    one, and making it sometimes-a-list would put a type check in all of them.
  * the wire accepts the standard OpenAI list shape, because that is what every
    OpenAI-compatible client already sends.
  * parts WIN over content at the provider boundary, or the image is silently
    dropped and the model is asked to read a page it was never shown.
"""

from __future__ import annotations

from src.domain.llm.value_objects import LLMMessage, MessageRole
from src.infrastructure.llm.litellm_provider import _to_litellm_messages
from src.interface.http.llm_router import LLMCompleteMessage

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
TEXT = {"type": "text", "text": "Read the invoice total."}


def test_a_text_message_is_unchanged():
    """The path every existing call takes. If this shifts, every prompt in the
    fleet shifts with it."""
    out = _to_litellm_messages([LLMMessage(role=MessageRole.USER, content="hello")])
    assert out == [{"role": "user", "content": "hello"}]


def test_a_page_image_reaches_the_provider_as_content_parts():
    out = _to_litellm_messages([LLMMessage(role=MessageRole.USER, parts=(TEXT, IMAGE))])
    assert out[0]["content"] == [TEXT, IMAGE]


def test_parts_win_over_content():
    """🚨 Both set, the image must not be dropped. Sending the string instead
    asks the model to read a page it was never shown — and it will answer,
    confidently, from the prompt alone."""
    out = _to_litellm_messages([LLMMessage(role=MessageRole.USER, content="fallback text", parts=(TEXT, IMAGE))])
    assert out[0]["content"] == [TEXT, IMAGE]


def test_an_empty_parts_tuple_falls_back_to_the_text():
    out = _to_litellm_messages([LLMMessage(role=MessageRole.USER, content="hi", parts=())])
    assert out[0]["content"] == "hi"


def test_a_text_message_is_still_hashable():
    """Frozen dataclass. Parts hold dicts, so a multimodal message is not
    hashable — every existing message is text and must stay so."""
    assert hash(LLMMessage(role=MessageRole.USER, content="hello"))


def test_tool_calls_still_travel_alongside():
    from src.domain.llm.value_objects import LLMToolCall

    out = _to_litellm_messages(
        [
            LLMMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(LLMToolCall(id="c1", name="lookup", arguments='{"a":1}'),),
            )
        ]
    )
    assert out[0]["tool_calls"][0]["function"]["name"] == "lookup"


# ── the wire ─────────────────────────────────────────────────────────────────


def test_the_wire_accepts_a_plain_string():
    assert LLMCompleteMessage(role="user", content="hello").content == "hello"


def test_the_wire_accepts_openai_content_parts():
    """The shape every OpenAI-compatible client already writes. Refusing it
    would mean each caller needs a special case for this one service."""
    assert LLMCompleteMessage(role="user", content=[TEXT, IMAGE]).content == [TEXT, IMAGE]
