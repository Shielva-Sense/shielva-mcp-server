"""Cross-context shared kernel: tenant identity, ids, base errors."""

from .errors import ConflictError, DomainError, NotFoundError, UnauthorizedError
from .tenant import TenantContext

__all__ = [
    "ConflictError",
    "DomainError",
    "NotFoundError",
    "TenantContext",
    "UnauthorizedError",
]
