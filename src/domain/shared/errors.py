"""Base domain exceptions. Adapters translate these into HTTP / JSON-RPC."""
from __future__ import annotations


class DomainError(Exception):
    """Root of every domain-raised exception.

    Application services catch ``DomainError`` and translate it into
    the interface's error shape (HTTP status, JSON-RPC error code).
    Domain code never imports interface modules.
    """


class NotFoundError(DomainError):
    """Aggregate root or value object not found in the repository."""


class ConflictError(DomainError):
    """Invariant violation — state would conflict with what the
    aggregate already says (e.g. session already initialized)."""


class UnauthorizedError(DomainError):
    """Caller lacks permission to perform the operation. Distinct
    from "not authenticated at all" (which is an interface concern)."""
