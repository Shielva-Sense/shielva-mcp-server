"""Infrastructure layer — driven adapters.

Each adapter implements a port (interface) defined under
``domain/``. Examples:

    domain/chat/repositories.py        ChatSessionRepository (port)
    infrastructure/persistence/        InMemoryChatSessionRepository (adapter)

    domain/llm/repositories.py         LLMProvider (port)
    infrastructure/llm/                LiteLLMProvider (adapter)

Adapters MAY import:
    * The framework they wrap (litellm, motor, httpx, etc.)
    * The port they implement (from domain/)
    * Shared kernel types (TenantContext, errors)

Adapters MUST NOT import:
    * Other adapters in infrastructure/ (talk through the domain)
    * application/ or interface/
"""
