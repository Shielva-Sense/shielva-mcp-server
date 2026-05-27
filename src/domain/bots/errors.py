"""Bots-context domain errors."""
from __future__ import annotations

from ..shared.errors import NotFoundError


class BotNotFoundError(NotFoundError):
    """Bot id unknown OR not visible to the calling tenant.

    The interface adapter must NOT distinguish these two cases —
    leaking "exists but not yours" is a cross-tenant info leak.
    """
