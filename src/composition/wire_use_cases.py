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
    app: Any,  # FastAPI — typed as Any to avoid framework import at module load
    *,
    tool_registry: Any,
    kb_registry: Any,
    bot_registry: Any,
    policy_engine: Any,
    context_assembler: Any,
    llm_router: Any,
    embedding_client: Any,
    vector_store: Any,
    rag_client: Any,
) -> None:
    """Wire the DDD/hexagonal layer onto a running FastAPI app.

    Call this AFTER the legacy infrastructure is constructed and
    populated (i.e. after ``register_codegen_tools(tool_registry)``
    + the rest of the lifespan's startup code) so the adapters see
    the final state of each registry.
    """
    # ── Slice 2: tools ──────────────────────────────────────────
    from src.application.tools import ToolApplicationService
    from src.infrastructure.tools import LegacyToolRegistryAdapter

    tool_adapter = LegacyToolRegistryAdapter(tool_registry)

    # 🚨 A tenant's installed connectors, as tools, BESIDE the built-ins.
    #
    # Built-ins are registered once at startup and are the same for everybody.
    # Connector tools are neither: this workspace has Gmail installed, the next
    # has nothing, and the set has to be read per request. That is why they
    # arrive as a second source rather than as more registrations — a registry
    # populated at boot has no tenant to scope itself to.
    #
    # The legacy adapter is FIRST, so a connector cannot shadow a built-in by
    # being named after it. Composition degrades one source at a time: if the
    # connector runtime is unreachable the built-in tools still list.
    #
    # Off unless a runtime URL is configured, so a deployment that has no
    # connector runtime is unchanged rather than logging a failure per request.
    from config.settings import get_settings

    _settings = get_settings()
    sources = [(tool_adapter, tool_adapter)]
    _connector_url = (getattr(_settings, "connector_gateway_url", "") or "").strip()
    if _connector_url:
        from src.infrastructure.tools.composite_catalogue import CompositeToolCatalogue
        from src.infrastructure.tools.connector_catalogue import ConnectorToolCatalogue

        _connectors = ConnectorToolCatalogue(
            base_url=_connector_url,
            timeout_s=float(getattr(_settings, "connector_timeout_seconds", 30) or 30),
        )
        sources.append((_connectors, _connectors))
        catalogue: Any = CompositeToolCatalogue(sources)
    else:
        catalogue = tool_adapter

    app.state.tool_app_service = ToolApplicationService(
        catalogue=catalogue,
        executor=catalogue,
    )

    # ── Slice 3: knowledge + LLM ────────────────────────────────
    from src.application.knowledge import KnowledgeApplicationService
    from src.application.llm import LLMApplicationService
    from src.infrastructure.llm import LiteLLMProviderAdapter
    from src.infrastructure.persistence import LegacyKBRepositoryAdapter
    from src.infrastructure.retrieval import (
        LegacyEmbeddingClientAdapter,
        LegacyHybridRetrieverAdapter,
        PgVectorStoreAdapter,
    )

    embedding_client_port = LegacyEmbeddingClientAdapter(embedding_client)
    vector_store_port = PgVectorStoreAdapter(vector_store)
    retriever_port = LegacyHybridRetrieverAdapter(rag_client)
    kb_repository_port = LegacyKBRepositoryAdapter(kb_registry)
    llm_provider_port = LiteLLMProviderAdapter(llm_router)

    app.state.knowledge_app_service = KnowledgeApplicationService(
        kb_repository=kb_repository_port,
        retriever=retriever_port,
        embedding_client=embedding_client_port,
    )
    app.state.llm_app_service = LLMApplicationService(
        provider=llm_provider_port,
    )
    # Raw ports — exposed so slice 4c+ callers can use the typed
    # adapter directly instead of touching app.state.<legacy>.
    app.state.embedding_client_port = embedding_client_port
    app.state.vector_store_port = vector_store_port
    app.state.retriever_port = retriever_port
    app.state.kb_repository_port = kb_repository_port
    app.state.llm_provider_port = llm_provider_port

    # ── Slice 4a: bots + policy ─────────────────────────────────
    from src.application.bots import BotApplicationService
    from src.application.policy import PolicyApplicationService
    from src.infrastructure.persistence import LegacyBotRepositoryAdapter
    from src.infrastructure.policy import OPAPolicyEngineAdapter

    bot_repository_port = LegacyBotRepositoryAdapter(bot_registry)
    policy_engine_port = OPAPolicyEngineAdapter(policy_engine)
    app.state.bot_app_service = BotApplicationService(repository=bot_repository_port)
    app.state.policy_app_service = PolicyApplicationService(engine=policy_engine_port)
    app.state.bot_repository_port = bot_repository_port
    app.state.policy_engine_port = policy_engine_port

    # ── HandleQuery use case — owns the query pipeline orchestration ─
    # (context assembly -> per-bot tools -> LLM+tool loop). No longer a
    # shim over protocol.MessageHandler; the DDD layer is the real path.
    from src.application.chat import HandleQueryUseCase

    app.state.handle_query_use_case = HandleQueryUseCase(
        context_assembler=context_assembler,
        tool_registry=tool_registry,
        llm_router=llm_router,
        # Token-streaming provider for /query/stream. Only used for bots with
        # no tools enabled — the provider's stream is text-only, so tool-enabled
        # bots keep the batched tool loop (see execute_stream).
        llm_provider=llm_provider_port,
        # 🚨 The connector source ONLY — not the composite.
        #
        # The composite also carries the built-in tools, and `get_tools_for_bot`
        # has already returned those. Handing the composite here would offer
        # every built-in twice, and a duplicated tool name in a prompt is a
        # model picking between two identical options for no reason.
        #
        # None when no connector runtime is configured, which makes this whole
        # branch inert and leaves every bot exactly as it was.
        connector_catalogue=(_connectors if _connector_url else None),
    )

    # ── Slice 4c: generic LLM tool-calling loop ─────────────────
    # Composes LLMProvider + ToolCatalogue + ToolExecutor. Codegen
    # fix-agent migrates onto this so it doesn't have to touch
    # llm_router.execute() directly; future HandleQuery
    # decomposition consumes the same loop.
    from src.application.llm import CompleteWithToolLoopUseCase

    # 🚨 The COMPOSITE, so a connector tool the caller put in `input_.tools` can
    # actually be looked up and run. The loop does not list — tools are passed
    # in — so this widens what is CALLABLE, never what a model is offered, and
    # adds no upstream call to the turn. What a bot is offered is decided in
    # `handle_query`, per bot, opt-in.
    app.state.complete_with_tool_loop_use_case = CompleteWithToolLoopUseCase(
        provider=llm_provider_port,
        catalogue=catalogue,
        executor=catalogue,
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
