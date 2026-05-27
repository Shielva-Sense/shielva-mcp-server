"""Health / readiness / metrics probes.

These probes are NOT under the ``/mcp/v1/...`` prefix — they're
operational surfaces that orchestrators (k8s, gateway health checks)
poll on the bare path. Kept here for visibility into what the
service exposes; no auth (probes are infra-level).
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Request, Response

from config.settings import get_settings

_settings = get_settings()

health_router = APIRouter(tags=["health"])


@health_router.get("/health")
async def health() -> dict:
    return {
        "status":  "healthy",
        "service": "mcp-server",
        "version": _settings.app_version,
    }


@health_router.get("/ready")
async def ready(request: Request) -> dict:
    """Reports presence of each new-layer application service on
    app.state. Legacy infra-singletons are no longer checked here —
    the new ports are the authoritative readiness signal."""
    state = request.app.state
    return {
        "status": "ready",
        "components": {
            "tool_app_service":      hasattr(state, "tool_app_service"),
            "knowledge_app_service": hasattr(state, "knowledge_app_service"),
            "llm_app_service":       hasattr(state, "llm_app_service"),
            "bot_app_service":       hasattr(state, "bot_app_service"),
            "policy_app_service":    hasattr(state, "policy_app_service"),
            "handle_query_use_case": hasattr(state, "handle_query_use_case"),
        },
    }


@health_router.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    """Scrape endpoint for sop_sdk's HTTP-level metrics middleware.

    sop_sdk's ``_install_http_metrics`` registers counters/histograms
    into the prometheus_client default registry; this endpoint just
    exposes them in scrape format. No custom counters live here —
    new telemetry goes through sop_sdk's API, not raw
    prometheus_client (see the durable rule in memory)."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return Response(
            status_code=404,
            content=b"prometheus_client not installed",
        )
