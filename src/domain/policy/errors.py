"""Policy-context errors."""

from __future__ import annotations

from ..shared.errors import UnauthorizedError


class PolicyDeniedError(UnauthorizedError):
    """OPA (or the fallback) returned ``allowed=False``.

    Application services that *require* policy approval (handle_query,
    provision_bot) raise this; the interface layer maps to
    JSON-RPC ``-31998 Forbidden`` or HTTP 403 depending on the
    transport.
    """
