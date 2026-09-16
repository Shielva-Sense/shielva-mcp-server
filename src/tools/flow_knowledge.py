"""What a model has to KNOW before it can build a flow, as a tool it can read.

WHY THIS IS A TOOL AND NOT A DOCUMENT

An MCP client learns a server from ``tools/list`` and nothing else — no README,
no schema registry, no codebase. So everything a model needs in order to build a
working flow has to arrive through a tool call. Without this it guesses node
kinds, and a flow full of invented kinds saves cleanly and then does nothing at
runtime, because ``flow_runtime`` simply never matches them.

🚨 THE ORDERING RULE IS THE POINT OF THIS FILE.

Four node kinds do something OUTSIDE the conversation — sms, mail, calendar,
crm — and each is inert unless the matching capability is enabled on the bot
first. This is not a style preference: the runtime's own comment records that
ARC once offered all four in the palette while the runtime knew none of them, so
authors dragged in nodes that silently did nothing. A model building a flow will
reproduce exactly that failure unless it is told, in the tool it reads before
building, that the capability comes first.

The same shape applies to ``action`` (needs an action schema to point at) and to
decision rules (need the signals they test to exist).

🚨 KEPT IN STEP WITH THE RUNTIME BY A TEST, not by care. ``_CAPABILITY_OF`` in
core-api's ``flow_runtime`` is the authority; the test asserts this table names
the same four. A copy that drifts is worse than no copy, because a model trusts
it.
"""

from __future__ import annotations

from typing import Any

import structlog

from src.protocol.models import TenantContext, ToolDefinition

logger = structlog.get_logger(__name__)

#: Node kinds, taken from ARC's palette and core-api's flow_runtime. `requires`
#: names what must exist BEFORE a node of this kind will do anything.
NODE_KINDS: list[dict[str, Any]] = [
    {"kind": "message", "label": "Message", "does": "the bot says something", "key_fields": ["text"]},
    {
        "kind": "ask",
        "label": "Ask",
        "does": "collect one answer from the caller into a variable",
        "key_fields": ["text", "field"],
    },
    {
        "kind": "classify",
        "label": "Classify",
        "does": "route by intent — the main branching node",
        "key_fields": ["cases"],
        "requires": "the intents it routes on must exist (shielva_create_intent)",
    },
    {
        "kind": "condition",
        "label": "Condition",
        "does": "branch on the value of a variable",
        "key_fields": ["conditions"],
    },
    {
        "kind": "branch",
        "label": "Branch",
        "does": "keyword or yes/no pick",
        "key_fields": ["branches"],
    },
    {
        "kind": "answer",
        "label": "Answer",
        "does": "answer from the knowledge base or free-form",
        "key_fields": ["source"],
        "requires": "a knowledge base linked to the bot, for source='kb'",
    },
    {
        "kind": "action",
        "label": "Action",
        "does": "call a connector",
        "key_fields": ["actionId"],
        "requires": "an action schema to point at (shielva_create_action_schema), and the connector installed",
    },
    {
        "kind": "agent",
        "label": "Agent",
        "does": "multi-step tool use toward a goal",
        "key_fields": ["goal", "tools", "maxIterations"],
    },
    {
        "kind": "painter",
        "label": "Render",
        "does": "render a card or table into the conversation",
        "key_fields": ["painterType", "painterDataSource", "painterDynamicFields"],
        "requires": "a painter to render with, unless painterStandalone is true (shielva_create_painter)",
    },
    {"kind": "set", "label": "Set", "does": "assign variables", "key_fields": ["assignments"]},
    {
        "kind": "data",
        "label": "Data",
        "does": "declare a structured object or list the flow works with",
        "key_fields": ["dataKind", "dataFields"],
    },
    {
        "kind": "loop",
        "label": "Loop",
        "does": "iterate over a list, or retry",
        "key_fields": ["loopType", "loopSource", "itemVar", "maxIterations"],
    },
    {"kind": "handoff", "label": "Handoff", "does": "hand the conversation to a human", "key_fields": []},
    {"kind": "subflow", "label": "Subflow", "does": "call a reusable sub-graph", "key_fields": ["flowId"]},
    {"kind": "collect", "label": "Collect", "does": "gather several fields in one step", "key_fields": []},
    {"kind": "wait", "label": "Wait", "does": "pause before continuing", "key_fields": []},
    {"kind": "pace", "label": "Pace", "does": "control speaking pace on a voice call", "key_fields": []},
    {"kind": "button", "label": "Button", "does": "offer tappable choices on a chat channel", "key_fields": []},
    {"kind": "route", "label": "Route", "does": "send the conversation down a named path", "key_fields": []},
    {"kind": "filter", "label": "Filter", "does": "narrow a list", "key_fields": []},
    {"kind": "map", "label": "Map", "does": "transform each item of a list", "key_fields": []},
    {"kind": "sort", "label": "Sort", "does": "order a list", "key_fields": []},
    {"kind": "automap", "label": "Auto-map", "does": "map connector output onto flow variables", "key_fields": []},
    {"kind": "start", "label": "Start", "does": "the entry node", "key_fields": []},
    {"kind": "startover", "label": "Start over", "does": "restart the conversation", "key_fields": []},
]

#: 🚨 MIRRORS ``_CAPABILITY_OF`` in core-api ``app/services/flow_runtime.py``.
#: A node of these kinds is INERT until the capability is enabled on the bot.
#: The runtime's own comment records the bug this prevents: the palette offered
#: all four while the runtime knew none, so authors dragged in nodes that did
#: nothing and nothing said why.
CAPABILITY_OF_KIND: dict[str, str] = {
    "sms": "sms.send",
    "mail": "mail.send",
    "calendar": "calendar.create_event",
    "crm": "crm.create_lead",
}

#: The order things must be done in. Stated as steps because a model reads this
#: once and then acts; a prose paragraph invites it to start at step four.
BUILD_ORDER: list[dict[str, str]] = [
    {
        "step": "1",
        "do": "Configure the bot FIRST",
        "why": (
            "Enable the capability a node needs (shielva_set_bot_capability) and define the signals "
            "it emits (shielva_set_bot_signals). A capability node added before its capability is "
            "enabled saves fine and then does nothing at runtime, with no error anywhere."
        ),
    },
    {
        "step": "2",
        "do": "Create what the nodes will point at",
        "why": (
            "Intents for classify nodes, an action schema for action nodes, a painter for painter "
            "nodes, a knowledge base for answer nodes. A node referencing something that does not "
            "exist is a dead branch."
        ),
    },
    {
        "step": "3",
        "do": "Add flow nodes one at a time",
        "why": (
            "shielva_add_flow_node saves after each node, so somebody watching the dialog designer "
            "sees them appear. Pass expected_node_count so a concurrent edit is reported rather "
            "than overwritten."
        ),
    },
    {
        "step": "4",
        "do": "Test before deploying",
        "why": "Use the bot's test surface; deploying publishes to live channels.",
    },
]


async def shielva_flow_guide(tenant_context: TenantContext) -> dict[str, Any]:
    """Everything needed to build a working flow: node kinds, prerequisites, order."""
    return {
        "status": "ok",
        "build_order": BUILD_ORDER,
        "node_kinds": NODE_KINDS,
        "capability_nodes": [
            {
                "kind": kind,
                "capability": cap,
                "enable_with": "shielva_set_bot_capability(bot_id, capability, enabled=True)",
                "if_skipped": "the node saves and then does nothing at runtime — no error is raised",
            }
            for kind, cap in CAPABILITY_OF_KIND.items()
        ],
        "note": (
            "Node kinds not in node_kinds are not recognised by the runtime. A flow containing one "
            "saves cleanly and that branch never executes."
        ),
    }


FLOW_KNOWLEDGE_TOOL_DEFINITIONS: list[tuple[ToolDefinition, Any]] = [
    (
        ToolDefinition(
            name="shielva_flow_guide",
            description=(
                "READ THIS FIRST before building or editing a dialog flow. Returns every node kind "
                "the runtime recognises, what each does, which fields it needs, and — importantly — "
                "what must be configured BEFORE a node will work: sms/mail/calendar/crm nodes are "
                "inert until the matching capability is enabled on the bot, action nodes need an "
                "action schema, classify nodes need intents, painter nodes need a painter. A node "
                "added out of order saves cleanly and then does nothing at runtime."
            ),
            parameters=[],
            requires_permissions=[],
            enabled_by_default=False,
        ),
        shielva_flow_guide,
    ),
]


def register_flow_knowledge_tools(registry) -> None:
    for definition, handler in FLOW_KNOWLEDGE_TOOL_DEFINITIONS:
        registry.register(definition, handler)
    logger.info("flow_knowledge_tools_registered", count=len(FLOW_KNOWLEDGE_TOOL_DEFINITIONS))
