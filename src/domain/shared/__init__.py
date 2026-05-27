"""Cross-context shared kernel: tenant identity, ids, base errors."""
from .errors import DomainError, NotFoundError, ConflictError, UnauthorizedError
from .tenant import TenantContext

__all__ = [
    "DomainError", "NotFoundError", "ConflictError", "UnauthorizedError",
    "TenantContext",
]
