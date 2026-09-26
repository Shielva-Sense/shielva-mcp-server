"""How a knowledge answer is written — per channel, and in the user's language.

Each test names what a customer would see if it broke: `<p>` tags in a
WhatsApp message, an English answer to an Arabic question.
"""

from __future__ import annotations

import asyncio

import pytest

from src.context.assembler import ContextAssembler
from src.protocol.models import TenantContext

PERSONA = "You are the front desk of a clinic. When they write Arabic, reply in Egyptian Arabic."


def _prompt(channel: str) -> str:
    assembler = ContextAssembler(rag_client=None, bot_registry=None, session_store=None, prompt_engine=None)
    tenant = TenantContext(tenant_id="Tenant-x", user_id="u1", user_email="u@example.com")
    bot = {"prompt_config": {"system_prompt": PERSONA}}
    return asyncio.run(assembler._build_system_prompt(bot_config=bot, tenant_context=tenant, channel=channel))


@pytest.mark.parametrize("channel", ["chat", "text", "voice", "something-new"])
def test_every_channel_answers_in_the_users_language(channel):
    prompt = _prompt(channel)
    assert "Reply in the language of the user's latest message" in prompt
    # The bot's own instruction still comes first — the rule defers to it for dialect.
    assert prompt.index(PERSONA) < prompt.index("Reply in the language")


def test_a_messaging_app_gets_plain_text_not_html():
    """WhatsApp shows `<p>` as the three characters it is."""
    prompt = _prompt("text")
    assert "Plain text only" in prompt
    assert "valid HTML" not in prompt


def test_the_web_widget_still_gets_html():
    assert "valid HTML" in _prompt("chat")
    assert "valid HTML" in _prompt("something-new"), "unknown channels keep the old behaviour"


def test_no_english_example_answer_to_copy():
    """An English example ("Yes, I found some messages from…") pulled Arabic answers into English."""
    for channel in ("chat", "text", "voice"):
        assert "I found some messages" not in _prompt(channel)
