"""The calling client's own credential, for the length of one request.

WHY THIS EXISTS, AND WHY IT IS NOT A FIELD ON ``TenantContext``

A flow tool has to call back into the platform — ``GET /bots/{id}/flow``,
``POST /bots/{id}/flow``. The question is what it authenticates as, and there is
only one answer that is not a security regression.

🚨 IT MUST CALL AS THE CALLER, NOT AS A SERVICE.

A tool holding a service token would reach every bot in every workspace, and the
whole MCP grant model — which routes and paths a key may touch, enforced by the
gateway's ``auth.grants`` plugin — would be bypassed the moment the request
originated inside the MCP server instead of at the gateway. The grants would
still be *configured*, and would protect nothing, which is the worst state to be
in: a control surface that reads as enforced.

So a tool re-presents the credential the client sent, and the gateway makes the
same decision it would have made for a direct call. One enforcement point. The
gateway already forwards ``Authorization`` / ``X-API-Key`` to upstreams (they
are absent from its identity strip list), so the credential is here to be used.

🚨 A CONTEXTVAR, NOT A MODEL FIELD. ``TenantContext`` is passed to every handler
and appears in log lines and error payloads across this service. Putting a live
credential on it would be one ``logger.info(tenant_context=...)`` away from
printing customer credentials into the log stream. A ContextVar is request
scoped, never serialised with the context object, and has exactly one reader.
"""

from __future__ import annotations

from contextvars import ContextVar

#: Headers worth carrying. ``Authorization`` covers the MCP OAuth path,
#: ``X-API-Key`` the minted-key path; a client uses one or the other.
_FORWARDABLE = ("authorization", "x-api-key")

_caller_credential: ContextVar[dict[str, str] | None] = ContextVar(
    "shielva_caller_credential",
    default=None,
)


def remember_caller_credential(headers: object) -> None:
    """Capture the caller's credential for this request.

    Called by the transport as a request arrives. Silently does nothing when no
    credential is present — an unauthenticated path simply has none to forward,
    and the upstream will refuse on its own rather than here.
    """
    found: dict[str, str] = {}
    for name in _FORWARDABLE:
        try:
            value = headers.get(name)  # type: ignore[attr-defined]
        except Exception:
            value = None
        if value:
            # Canonical casing on the way out; HTTP is case-insensitive but some
            # upstream middlewares are not as careful as they should be.
            found["Authorization" if name == "authorization" else "X-API-Key"] = value
    _caller_credential.set(found or None)


def caller_auth_headers() -> dict[str, str]:
    """The credential to re-present upstream, or ``{}`` when there is none.

    🚨 Returns a COPY. A caller that mutates the result — adding a header, say —
    must not thereby edit what every later tool in the same request forwards.
    """
    return dict(_caller_credential.get() or {})


def has_caller_credential() -> bool:
    return bool(_caller_credential.get())
