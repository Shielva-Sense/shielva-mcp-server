"""Knowledge-context domain errors."""
from __future__ import annotations

from ..shared.errors import NotFoundError


class KnowledgeBaseNotFoundError(NotFoundError):
    """KB id is unknown OR not visible to the calling tenant.

    The interface adapter never distinguishes these two cases on the
    wire — leaking "exists but not yours" is a cross-tenant info leak.
    """
