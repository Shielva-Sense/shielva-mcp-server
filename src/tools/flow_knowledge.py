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
    {
        "kind": "mail",
        "label": "Mail",
        "does": "send an email through a connector — NO action schema needed",
        "key_fields": ["capConnector", "capAction", "capFields"],
        "requires": "the mail.send capability enabled on the bot (shielva_set_bot_capability)",
    },
    {
        "kind": "sms",
        "label": "SMS",
        "does": "send a text through a connector — NO action schema needed",
        "key_fields": ["capConnector", "capAction", "capFields"],
        "requires": "the sms.send capability enabled on the bot",
    },
    {
        "kind": "calendar",
        "label": "Calendar",
        "does": "create an event through a connector — NO action schema needed",
        "key_fields": ["capConnector", "capAction", "capFields"],
        "requires": "the calendar.create_event capability enabled on the bot",
    },
    {
        "kind": "crm",
        "label": "CRM",
        "does": "create a lead through a connector — NO action schema needed",
        "key_fields": ["capConnector", "capAction", "capFields"],
        "requires": "the crm.create_lead capability enabled on the bot",
    },
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


#: 🚨 TWO WAYS TO CALL A CONNECTOR, and picking the wrong one is the difference
#: between a flow that works and an afternoon spent authoring a schema nobody
#: needed. The owner's rule, which matches what the runtime does:
#:
#:   Just EXECUTE the API (send the mail, create the lead)
#:       → capability node. NO action schema. The node names the connector and
#:         action itself and carries canonical fields.
#:
#:   RENDER the response (show a card or a table of what came back)
#:       → action schema, then a painter. A `static` painter renders from a
#:         chosen action schema's RESPONSE SHAPE — that shape is what the schema
#:         exists to describe, and without it there is nothing to build columns
#:         from.
#:
#: 🚨 CANONICAL FIELD NAMES, NEVER THE VENDOR'S. A capability node stores `to`,
#: `subject`, `body`; each connector declares a `map` in its `capability_actions`
#: saying where those land (`to`→`dst`, `body`→`text`). That translation is the
#: entire reason these nodes exist: it lets a workspace swap Twilio for Plivo, or
#: Gmail for Outlook, without re-typing a single field in the flow. Sending
#: vendor names straight through reaches the provider with parameters it has
#: never heard of — a 400 that reads like a credentials problem.
CAPABILITY_NODE_FIELDS: dict[str, Any] = {
    "capConnector": "Connector TYPE to call, e.g. google_gmail_connector. Omit to use the bot's configured provider.",
    "capAction": "The connector's own action name, e.g. send_email. From the connector's capability_actions.",
    "capFields": "CANONICAL field names only — to, subject, body. The connector maps them to its own params.",
    "capOpaque": "true when the provider takes ONE opaque object instead of per-field values.",
    "capPayload": "The raw JSON body, used only when capOpaque is true.",
    "capResultVar": "Variable to store the call's result in, if a later node needs it.",
    "waitingMessage": "Said BEFORE the call on a voice channel — a connector round-trip is seconds of silence.",
    "completionMessage": "Said after, and may reference what the call returned.",
}

#: The four kinds that take the fields above.
CAPABILITY_NODE_KINDS: list[dict[str, Any]] = [
    {"kind": "mail", "capability": "mail.send", "canonical_fields": ["to", "subject", "body"]},
    {"kind": "sms", "capability": "sms.send", "canonical_fields": ["to", "body"]},
    {
        "kind": "calendar",
        "capability": "calendar.create_event",
        "canonical_fields": ["title", "start", "end", "attendees"],
    },
    {"kind": "crm", "capability": "crm.create_lead", "canonical_fields": ["name", "email", "phone", "company"]},
]

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
        "calling_a_connector": {
            "just_execute_it": {
                "use": "a capability node (mail, sms, calendar, crm)",
                "action_schema_needed": False,
                "how": (
                    "Enable the capability on the bot, then add the node with capConnector, "
                    "capAction and capFields. Sending an email needs NO action schema."
                ),
                "node_fields": CAPABILITY_NODE_FIELDS,
                "kinds": CAPABILITY_NODE_KINDS,
                "canonical_fields_note": (
                    "Use canonical names (to, subject, body) — never the vendor's parameter names. "
                    "The connector maps them, which is what lets the provider be swapped without "
                    "re-authoring the flow."
                ),
            },
            "render_the_response": {
                "use": "an action node pointing at an action schema, then a painter",
                "action_schema_needed": True,
                "how": (
                    "Create the action schema first (shielva_create_action_schema): a static painter "
                    "builds its card or table from that schema's RESPONSE SHAPE, so without it there "
                    "is nothing to render from."
                ),
            },
        },
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
