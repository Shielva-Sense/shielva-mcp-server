"""Three-tier domain exception hierarchy for shielva-mcp."""

from __future__ import annotations


class ShielvaException(Exception):
    """Base exception for all shielva-mcp domain errors."""

    status_code: int = 500
    retryable: bool = False
    error_code: str = "INTERNAL_ERROR"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class IntegrationException(ShielvaException):
    """Raised when an upstream integration (LLM, vector store, MongoDB) fails."""

    status_code = 502
    retryable = True
    error_code = "INTEGRATION_ERROR"


class RuntimeException(ShielvaException):
    """Raised for recoverable business-logic errors (bad input, not found, etc.)."""

    status_code = 400
    retryable = False
    error_code = "RUNTIME_ERROR"


class TechnicalException(ShielvaException):
    """Raised for unexpected internal failures that are not retryable."""

    status_code = 500
    retryable = False
    error_code = "TECHNICAL_ERROR"
