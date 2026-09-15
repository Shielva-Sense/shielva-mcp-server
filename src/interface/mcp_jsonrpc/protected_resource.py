"""RFC 9728 OAuth 2.0 protected-resource metadata for the MCP endpoint.

Why this exists
---------------
An MCP client (Claude Desktop) has no way to know that a "Sign in" flow is
possible unless the resource server tells it. RFC 9728 is that telling: the
client fetches a small JSON document from the resource and reads
``authorization_servers``. Absent the document there is no authorization
server to discover, so the only credential the client can offer the user is a
pasted API key.

Nothing here implements OAuth. shielva-identity is ALREADY a full OAuth 2.0 /
OIDC authorization server — PKCE S256 mandatory, single-use 5-minute
authorization codes, JWKS published (see shielva-identity
``app/api/v1/endpoints/oauth_provider.py`` and ``endpoints/jwks.py``). This
module only advertises it.

Public URLs — the gateway arithmetic
------------------------------------
From shielva-api-gateway ``config/routes.yaml``, rewrite is a plain prefix
strip (``regex_route_matcher._apply_rewrite``)::

    route `mcp`        path_prefix /api/mcp/    strip_prefix /api/mcp
      <gw>/api/mcp/.well-known/oauth-protected-resource
        → upstream /.well-known/oauth-protected-resource   ← this router

    route `identity`   path_prefix /identity/   strip_prefix /identity
      <gw>/identity/api/v1/oauth/authorize   → /api/v1/oauth/authorize
      <gw>/identity/api/v1/oauth/token       → /api/v1/oauth/token
      <gw>/identity/.well-known/jwks.json    → /.well-known/jwks.json

so the issuer identity advertises for itself is ``<gw>/identity`` — exactly
what its own discovery document returns (``settings.BACKEND_URL + "/identity"``
in ``jwks.py::oidc_discovery``). Both values are configuration here, never
literals: see ``MCP_OAUTH_RESOURCE_URL`` / ``MCP_OAUTH_AUTHORIZATION_SERVER``
in ``config/settings.py``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

# The path is fixed by RFC 9728 §3 — it is a protocol constant, not a setting.
WELL_KNOWN_PATH = "/.well-known/oauth-protected-resource"


def _settings() -> Any:
    """Read settings lazily.

    Deliberately not a module-level ``get_settings()``: this module is
    imported by the transport, and the transport is imported by tests that
    have no envelope. ``get_settings`` is ``lru_cache``d, so the call is free
    after the first one.
    """
    from config.settings import get_settings

    return get_settings()


def resource_url() -> str:
    """The public URL of the MCP resource, from config."""
    return _settings().mcp_oauth_resource_url.rstrip("/")


def authorization_server() -> str:
    """The public issuer base URL of shielva-identity, from config."""
    return _settings().mcp_oauth_authorization_server.rstrip("/")


def scopes_supported() -> list[str]:
    raw = _settings().mcp_oauth_scopes or ""
    return [s.strip() for s in raw.split(",") if s.strip()]


def resource_metadata_url() -> str:
    """Where the metadata document itself is served.

    This is what a 401's ``WWW-Authenticate`` must name, so it is derived
    from the same configured resource URL the document reports — the two can
    never disagree.
    """
    return resource_url() + WELL_KNOWN_PATH


def www_authenticate_value() -> str:
    """RFC 9728 §5.1 challenge.

    No ``error`` parameter: RFC 6750 §3.1 reserves those for a credential
    that was *presented and rejected*. This challenge answers a request that
    carried no credential at all, which is precisely the case the client must
    read as "go discover the authorization server".
    """
    return f'Bearer resource_metadata="{resource_metadata_url()}"'


def metadata_document() -> dict[str, Any]:
    """The RFC 9728 protected-resource metadata document."""
    return {
        "resource": resource_url(),
        "authorization_servers": [authorization_server()],
        "bearer_methods_supported": ["header"],
        "scopes_supported": scopes_supported(),
    }


def build_router() -> APIRouter:
    """Router serving the metadata at the service root.

    Mounted with no prefix so the path upstream is exactly
    ``/.well-known/oauth-protected-resource``; the gateway's ``/api/mcp``
    strip then makes the public URL ``<gw>/api/mcp`` + that path.
    """
    router = APIRouter(tags=["oauth-metadata"])

    @router.get(WELL_KNOWN_PATH)
    async def oauth_protected_resource() -> dict[str, Any]:
        return metadata_document()

    return router


__all__ = [
    "WELL_KNOWN_PATH",
    "authorization_server",
    "build_router",
    "metadata_document",
    "resource_metadata_url",
    "resource_url",
    "scopes_supported",
    "www_authenticate_value",
]
