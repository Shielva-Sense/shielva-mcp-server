"""
MCP Meeting Tools — TMS entity creation tools for post-meeting extraction.

Registered in mcp-server/src/main.py lifespan alongside codegen tools:
    from src.tools.meeting_tools import register_meeting_tools
    register_meeting_tools(tool_registry)

Tools:
  create_tms_goal          — create a Goal from meeting context
  create_tms_epic          — create an Epic linked to a Goal
  create_tms_sprint        — create a Sprint with dates
  create_tms_ticket        — create a Ticket for a specific action item
  meeting_context_query    — query the meeting transcript KB
  get_delegation_rules     — fetch the user's delegation persona config
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
import structlog

from src.protocol.models import TenantContext

logger = structlog.get_logger(__name__)

_TMS_URL = os.getenv("TMS_URL", "https://localhost:8002")
_TMS_TOKEN = os.getenv("TMS_TOKEN", "")
_PRESENCE_URL = os.getenv("PRESENCE_URL", "https://localhost:8060")
_SSL = False
_TIMEOUT = 15.0


def _tms_headers(tenant_id: str) -> dict:
    h = {"Content-Type": "application/json", "X-Tenant-Id": tenant_id}
    if _TMS_TOKEN:
        h["Authorization"] = f"Bearer {_TMS_TOKEN}"
    return h


async def _tms_post(path: str, tenant_id: str, payload: dict) -> dict:
    async with httpx.AsyncClient(verify=_SSL, timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{_TMS_URL}{path}",
            headers=_tms_headers(tenant_id),
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


# ── Tool handlers ─────────────────────────────────────────────────────────────

async def create_tms_goal(
    tenant_context: TenantContext,
    title: str,
    description: str = "",
    goal_type: str = "objective",
    created_by: str = "meeting-bot",
) -> Dict[str, Any]:
    """
    Create a strategic Goal in TMS from meeting transcript context.
    Use for high-level quarterly or annual objectives discussed in the meeting.
    """
    try:
        result = await _tms_post(
            "/tms/api/v1/goals/",
            tenant_context.tenant_id,
            {
                "title": title,
                "description": description,
                "status": "on_track",
                "goal_type": goal_type,
                "progress": 0,
                "created_by": created_by,
            },
        )
        logger.info("meeting_tool_goal_created", title=title, id=result.get("id"))
        return {"id": result.get("id"), "title": title, "status": "created"}
    except Exception as e:
        logger.error("meeting_tool_goal_failed", title=title, error=str(e))
        return {"error": str(e), "title": title, "status": "failed"}


async def create_tms_epic(
    tenant_context: TenantContext,
    title: str,
    description: str = "",
    goal_id: Optional[str] = None,
    priority: str = "medium",
    created_by: str = "meeting-bot",
) -> Dict[str, Any]:
    """
    Create an Epic in TMS. Link to a Goal by goal_id if one was just created.
    Use for major work streams or feature areas discussed in the meeting.
    """
    payload: dict = {
        "title": title,
        "description": description,
        "status": "backlog",
        "priority": priority,
        "progress": 0,
        "created_by": created_by,
    }
    if goal_id:
        payload["goal_id"] = goal_id

    params = "?auto_parent=true" if not goal_id else ""
    try:
        result = await _tms_post(
            f"/tms/api/v1/epics/{params}",
            tenant_context.tenant_id,
            payload,
        )
        logger.info("meeting_tool_epic_created", title=title, id=result.get("id"))
        return {"id": result.get("id"), "title": title, "status": "created"}
    except Exception as e:
        logger.error("meeting_tool_epic_failed", title=title, error=str(e))
        return {"error": str(e), "title": title, "status": "failed"}


async def create_tms_sprint(
    tenant_context: TenantContext,
    name: str,
    goal: str,
    start_date: str,
    end_date: str,
    epic_id: Optional[str] = None,
    created_by: str = "meeting-bot",
) -> Dict[str, Any]:
    """
    Create a Sprint in TMS from a time-boxed iteration discussed in the meeting.
    start_date and end_date must be ISO date strings (YYYY-MM-DD).
    """
    payload: dict = {
        "name": name,
        "goal": goal,
        "start_date": start_date,
        "end_date": end_date,
        "status": "planning",
        "created_by": created_by,
    }
    if epic_id:
        payload["epic_id"] = epic_id

    params = "?auto_parent=true" if not epic_id else ""
    try:
        result = await _tms_post(
            f"/tms/api/v1/sprints/{params}",
            tenant_context.tenant_id,
            payload,
        )
        logger.info("meeting_tool_sprint_created", name=name, id=result.get("id"))
        return {"id": result.get("id"), "name": name, "status": "created"}
    except Exception as e:
        logger.error("meeting_tool_sprint_failed", name=name, error=str(e))
        return {"error": str(e), "name": name, "status": "failed"}


async def create_tms_ticket(
    tenant_context: TenantContext,
    title: str,
    ticket_type: str = "task",
    description: str = "",
    priority: str = "MEDIUM",
    assigned_to_name: Optional[str] = None,
    sprint_id: Optional[str] = None,
    epic_id: Optional[str] = None,
    created_by: str = "meeting-bot",
) -> Dict[str, Any]:
    """
    Create a Ticket in TMS for a specific action item or task from the meeting.
    ticket_type must be one of: task, story, bug, feature.
    """
    payload: dict = {
        "title": title,
        "ticket_type": ticket_type,
        "description": description,
        "status": "BACKLOG",
        "priority": priority,
        "created_by": created_by,
    }
    if assigned_to_name:
        payload["assigned_to_name"] = assigned_to_name
    if sprint_id:
        payload["sprint_id"] = sprint_id
    if epic_id:
        payload["epic_id"] = epic_id

    try:
        result = await _tms_post(
            "/tms/api/v1/tickets/",
            tenant_context.tenant_id,
            payload,
        )
        logger.info("meeting_tool_ticket_created", title=title, id=result.get("id"))
        return {"id": result.get("id"), "title": title, "status": "created"}
    except Exception as e:
        logger.error("meeting_tool_ticket_failed", title=title, error=str(e))
        return {"error": str(e), "title": title, "status": "failed"}


async def meeting_context_query(
    tenant_context: TenantContext,
    query: str,
    meeting_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query the shielva-presence transcript store for context about a specific meeting.
    Use when you need to clarify a detail before creating a TMS entity.
    """
    try:
        async with httpx.AsyncClient(verify=_SSL, timeout=10.0) as client:
            params: dict = {}
            if meeting_id:
                resp = await client.get(
                    f"{_PRESENCE_URL}/presence/api/v1/transcripts/{meeting_id}",
                    headers={"X-Tenant-Id": tenant_context.tenant_id},
                )
                resp.raise_for_status()
                data = resp.json()
                turns = data.get("turns", [])
                context = "\n".join(
                    f"{t['speaker_name']}: {t['text']}" for t in turns[-20:]
                )
                return {"context": context, "turn_count": len(turns)}
            return {"context": "", "turn_count": 0}
    except Exception as e:
        return {"error": str(e), "context": ""}


async def get_delegation_rules(
    tenant_context: TenantContext,
    user_id: str,
) -> Dict[str, Any]:
    """
    Fetch the user's AI delegation persona rules (topic restrictions, language, KB IDs).
    Use to ensure bot responses respect the user's configured boundaries.
    """
    try:
        async with httpx.AsyncClient(verify=_SSL, timeout=10.0) as client:
            resp = await client.get(
                f"{_PRESENCE_URL}/presence/api/v1/delegation/persona",
                params={"user_id": user_id},
                headers={"X-Tenant-Id": tenant_context.tenant_id},
            )
            if resp.status_code == 404:
                return {"found": False, "topic_rules": [], "kb_ids": []}
            resp.raise_for_status()
            return {**resp.json(), "found": True}
    except Exception as e:
        return {"error": str(e), "found": False}


# ── Tool definitions (mirrors codegen_tools.py pattern) ──────────────────────

from src.protocol.models import ToolDefinition  # noqa: E402

MEETING_TOOL_DEFINITIONS = [
    (
        ToolDefinition(
            name="create_tms_goal",
            description=(
                "Create a strategic Goal in the TMS from meeting transcript content. "
                "Use for high-level objectives, OKRs, or quarterly targets discussed. "
                "Returns the created goal's id."
            ),
            parameters=[
                {"name": "title",       "type": "string", "description": "Goal title",              "required": True},
                {"name": "description", "type": "string", "description": "Goal description",         "required": False},
                {"name": "goal_type",   "type": "string", "description": "objective | key_result",   "required": False},
                {"name": "created_by",  "type": "string", "description": "User id of meeting host",  "required": False},
            ],
            requires_permissions=[],
            enabled_by_default=True,
        ),
        create_tms_goal,
    ),
    (
        ToolDefinition(
            name="create_tms_epic",
            description=(
                "Create an Epic in the TMS for a major work stream or feature area from the meeting. "
                "Optionally link to a Goal by goal_id. Returns the created epic's id."
            ),
            parameters=[
                {"name": "title",       "type": "string", "description": "Epic title",              "required": True},
                {"name": "description", "type": "string", "description": "Epic description",         "required": False},
                {"name": "goal_id",     "type": "string", "description": "Parent goal id",           "required": False},
                {"name": "priority",    "type": "string", "description": "low|medium|high|critical", "required": False},
                {"name": "created_by",  "type": "string", "description": "User id of meeting host",  "required": False},
            ],
            requires_permissions=[],
            enabled_by_default=True,
        ),
        create_tms_epic,
    ),
    (
        ToolDefinition(
            name="create_tms_sprint",
            description=(
                "Create a Sprint in the TMS for a time-boxed iteration discussed in the meeting. "
                "start_date and end_date are ISO date strings YYYY-MM-DD. "
                "Returns the created sprint's id."
            ),
            parameters=[
                {"name": "name",        "type": "string", "description": "Sprint name",              "required": True},
                {"name": "goal",        "type": "string", "description": "Sprint goal statement",    "required": True},
                {"name": "start_date",  "type": "string", "description": "ISO start date YYYY-MM-DD","required": True},
                {"name": "end_date",    "type": "string", "description": "ISO end date YYYY-MM-DD",  "required": True},
                {"name": "epic_id",     "type": "string", "description": "Parent epic id",           "required": False},
                {"name": "created_by",  "type": "string", "description": "User id of meeting host",  "required": False},
            ],
            requires_permissions=[],
            enabled_by_default=True,
        ),
        create_tms_sprint,
    ),
    (
        ToolDefinition(
            name="create_tms_ticket",
            description=(
                "Create a Ticket in the TMS for a specific action item or task from the meeting. "
                "ticket_type: task|story|bug|feature. Returns the created ticket's id."
            ),
            parameters=[
                {"name": "title",           "type": "string", "description": "Ticket title",                       "required": True},
                {"name": "ticket_type",     "type": "string", "description": "task|story|bug|feature",             "required": False},
                {"name": "description",     "type": "string", "description": "Ticket description",                  "required": False},
                {"name": "priority",        "type": "string", "description": "LOW|MEDIUM|HIGH|CRITICAL",            "required": False},
                {"name": "assigned_to_name","type": "string", "description": "Assignee display name from transcript","required": False},
                {"name": "sprint_id",       "type": "string", "description": "Parent sprint id",                    "required": False},
                {"name": "epic_id",         "type": "string", "description": "Parent epic id",                      "required": False},
                {"name": "created_by",      "type": "string", "description": "User id of meeting host",             "required": False},
            ],
            requires_permissions=[],
            enabled_by_default=True,
        ),
        create_tms_ticket,
    ),
    (
        ToolDefinition(
            name="meeting_context_query",
            description=(
                "Query the meeting transcript to get context for entity creation decisions. "
                "Returns recent speaker turns from the transcript."
            ),
            parameters=[
                {"name": "query",      "type": "string", "description": "What to look for",     "required": True},
                {"name": "meeting_id", "type": "string", "description": "The meeting id",       "required": False},
            ],
            requires_permissions=[],
            enabled_by_default=True,
        ),
        meeting_context_query,
    ),
    (
        ToolDefinition(
            name="get_delegation_rules",
            description=(
                "Fetch the user's AI persona delegation rules (topic restrictions, knowledge base IDs). "
                "Use before responding to ensure bot respects the user's configured boundaries."
            ),
            parameters=[
                {"name": "user_id", "type": "string", "description": "The Shielva user id", "required": True},
            ],
            requires_permissions=[],
            enabled_by_default=True,
        ),
        get_delegation_rules,
    ),
]


def register_meeting_tools(registry) -> None:
    """
    Register all meeting TMS tools into the MCP tool registry.

    Call from mcp-server/src/main.py lifespan:
        from src.tools.meeting_tools import register_meeting_tools
        register_meeting_tools(tool_registry)
    """
    for definition, handler in MEETING_TOOL_DEFINITIONS:
        registry.register(definition, handler)
    logger.info("meeting_tools_registered", count=len(MEETING_TOOL_DEFINITIONS))
