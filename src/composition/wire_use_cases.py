"""Wire the new-layer application services + ports onto ``app.state``.

This is the composition root for slices 1-4b. ``main.py`` constructs
the legacy infrastructure singletons (tool_registry, llm_router,
context_assembler, kb_registry, bot_registry, policy_engine,
message_handler, embedding_client, rag_client, vector_store) during
its lifespan and hands them in here as keyword arguments. We
construct the new-layer ports + application services on top, then
attach them to ``app.state`` for the interface adapters to consume.

Why a single ``wire_use_cases`` function (vs per-context functions):
    Reading order. A developer onboarding to the codebase sees one
    place that names every port + use case + which legacy
    infrastructure it wraps. The function is long but flat — no
    nested calls, no clever indirection.
"""
from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def wire_use_cases(
    app:                Any,  # FastAPI — typed as Any to avoid framework import at module load
    *,
    tool_registry:      Any,
    kb_registry:        Any,
    bot_registry:       Any,
    policy_engine:      Any,
    message_handler:    Any,
    llm_router:         Any,
    embedding_client:   Any,
    vector_store:       Any,
    rag_client:         Any,
) -> None:
    """Wire the DDD/hexagonal layer onto a running FastAPI app.

    Call this AFTER the legacy infrastructure is constructed and
    populated (i.e. after ``register_codegen_tools(tool_registry)``
    + the rest of the lifespan's startup code) so the adapters see
    the final state of each registry.
    """
    # ── Slice 2: tools ──────────────────────────────────────────
    from src.application.tools          import ToolApplicationService
    from src.infrastructure.tools       import LegacyToolRegistryAdapter
    tool_adapter = LegacyToolRegistryAdapter(tool_registry)
    app.state.tool_app_service = ToolApplicationService(
        catalogue = tool_adapter,
        executor  = tool_adapter,
    )

    # ── Slice 3: knowledge + LLM ────────────────────────────────
    from src.application.knowledge     import KnowledgeApplicationService
    from src.application.llm           import LLMApplicationService
    from src.infrastructure.llm        import LiteLLMProviderAdapter
    from src.infrastructure.persistence import LegacyKBRepositoryAdapter
    from src.infrastructure.retrieval  import (
        LegacyEmbeddingClientAdapter,
        LegacyHybridRetrieverAdapter,
        SupabaseVectorStoreAdapter,
    )
    embedding_client_port = LegacyEmbeddingClientAdapter(embedding_client)
    vector_store_port     = SupabaseVectorStoreAdapter(vector_store)
    retriever_port        = LegacyHybridRetrieverAdapter(rag_client)
    kb_repository_port    = LegacyKBRepositoryAdapter(kb_registry)
    llm_provider_port     = LiteLLMProviderAdapter(llm_router)

    app.state.knowledge_app_service = KnowledgeApplicationService(
        kb_repository    = kb_repository_port,
        retriever        = retriever_port,
        embedding_client = embedding_client_port,
    )
    app.state.llm_app_service = LLMApplicationService(
        provider = llm_provider_port,
    )
    # Raw ports — exposed so slice 4c+ callers can use the typed
    # adapter directly instead of touching app.state.<legacy>.
    app.state.embedding_client_port = embedding_client_port
    app.state.vector_store_port     = vector_store_port
    app.state.retriever_port        = retriever_port
    app.state.kb_repository_port    = kb_repository_port
    app.state.llm_provider_port     = llm_provider_port

    # ── Slice 4a: bots + policy ─────────────────────────────────
    from src.application.bots          import BotApplicationService
    from src.application.policy        import PolicyApplicationService
    from src.infrastructure.persistence import LegacyBotRepositoryAdapter
    from src.infrastructure.policy     import OPAPolicyEngineAdapter
    bot_repository_port = LegacyBotRepositoryAdapter(bot_registry)
    policy_engine_port  = OPAPolicyEngineAdapter(policy_engine)
    app.state.bot_app_service    = BotApplicationService(repository=bot_repository_port)
    app.state.policy_app_service = PolicyApplicationService(engine=policy_engine_port)
    app.state.bot_repository_port = bot_repository_port
    app.state.policy_engine_port  = policy_engine_port

    # ── Slice 4b: HandleQuery use case ──────────────────────────
    from src.application.chat          import HandleQueryUseCase
    app.state.handle_query_use_case = HandleQueryUseCase(
        legacy_message_handler = message_handler,
    )

    # ── Slice 4c: generic LLM tool-calling loop ─────────────────
    # Composes LLMProvider + ToolCatalogue + ToolExecutor. Codegen
    # fix-agent migrates onto this so it doesn't have to touch
    # llm_router.execute() directly; future HandleQuery
    # decomposition consumes the same loop.
    from src.application.llm import CompleteWithToolLoopUseCase
    app.state.complete_with_tool_loop_use_case = CompleteWithToolLoopUseCase(
        provider  = llm_provider_port,
        catalogue = tool_adapter,
        executor  = tool_adapter,
    )

    logger.info(
        "mcp.composition_wired",
        services=[
            "tool_app_service",
            "knowledge_app_service",
            "llm_app_service",
            "bot_app_service",
            "policy_app_service",
            "handle_query_use_case",
            "complete_with_tool_loop_use_case",
        ],
    )
