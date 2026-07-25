"""
Shielva MCP Server - Main Application
The AI Operating System for Shielva ARC
"""

# ── Envelope decryption (must run BEFORE any settings/env-reading imports) ──
import os as _envelope_os

_envelope_os.environ.setdefault("VAULT_SIDECAR_URL", "https://localhost:8054")
from dotenv import load_dotenv as _envelope_load_dotenv

_envelope_load_dotenv(".env", override=True)  # ciphertext + REDIS_URL passthrough
from shielva_common.envelope import bootstrap as _envelope_bootstrap

_envelope_bootstrap()
# ──────────────────────────────────────────────────────────────────────────

import json

# Configure logging
import logging
import os
import uuid
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from config.settings import get_settings
from src.context.assembler import ContextAssembler
from src.protocol.message_handler import MessageHandler
from src.registry.bot_registry import BotRegistry
from src.registry.kb_registry import KBRegistry
from src.registry.tool_registry import ToolRegistry, create_registry_with_defaults
from src.routing.llm_router import LLMRouter

settings = get_settings()

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(settings.log_level.upper())),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger(__name__)

# Global instances
tool_registry: ToolRegistry = None
llm_router: LLMRouter = None
context_assembler: ContextAssembler = None
message_handler: MessageHandler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global tool_registry, llm_router, context_assembler, message_handler

    logger.info("Starting MCP Server", version=settings.app_version)

    # Initialize components
    tool_registry = create_registry_with_defaults()

    llm_router = LLMRouter(
        tool_registry=tool_registry,
        default_model=settings.litellm_model,
        fallback_models=settings.litellm_fallback_models,
    )
    # Expose llm_router on app.state so codegen_routes can access it
    # without a circular import (routes cannot import from main.py).
    app.state.llm_router = llm_router
    # Also expose tool_registry on app.state for codegen fix-agent route
    app.state.tool_registry = tool_registry

    # Initialize OPA Policy Engine
    from src.security.policy_engine import OPAPolicyEngine

    policy_engine = OPAPolicyEngine(
        opa_url=settings.opa_url or "http://localhost:8181",
        policy_path="/v1/data/shielva",
    )

    # RAG Engine - pgvector (in-cluster Postgres)
    # We need to make sure src.rag_engine imports resolve correctly.
    # Assuming 'rag-engine/src' is in PYTHONPATH or symlinked?
    # Given project structure, we might need to adjust imports if not installed as package.
    # But let's assume standard import path based on structure.
    # Actually, the rag-engine seems to be separate.
    # I'll use local imports if needed or assume installed.
    # For now, I'll assume the files are importable.
    # Initialize MongoDB
    from motor.motor_asyncio import AsyncIOMotorClient

    from src.rag_engine.retriever import HybridRetriever
    from src.rag_engine.vectorstore import PgVectorStore

    try:
        # settings.mongodb_url may be a Pydantic SecretStr — unwrap before calling startswith
        _url_str = (
            settings.mongodb_url.get_secret_value()
            if hasattr(settings.mongodb_url, "get_secret_value")
            else str(settings.mongodb_url)
        )
        import certifi as _c

        _tls = {"tlsCAFile": _c.where()} if _url_str.startswith("mongodb+srv") else {}
    except ImportError:
        _tls = {}
        _url_str = (
            settings.mongodb_url.get_secret_value()
            if hasattr(settings.mongodb_url, "get_secret_value")
            else str(settings.mongodb_url)
        )
    mongo_client = AsyncIOMotorClient(_url_str, **_tls)

    # Initialize Vector Store
    # Use MCP_VECTOR_DB_URL (legacy SUPABASE_DB_URL alias) if available, else warn
    _raw_db_url = settings.mcp_vector_db_url
    # Unwrap Pydantic SecretStr — 'in' / split() / startswith() fail on SecretStr directly
    db_url = _raw_db_url.get_secret_value() if hasattr(_raw_db_url, "get_secret_value") else (_raw_db_url or "")
    if not db_url and settings.supabase_url:
        # Warn user or try to construct?
        # For now we rely on db_url being set in server.sh
        logger.warning("MCP_VECTOR_DB_URL not set. Vector store may fail to connect.")
        db_url = "postgresql://postgres:postgres@localhost:54322/postgres"  # Fallback/Invalid

    # DEBUG: Print DB URL (mask password)
    masked_url = db_url
    if "@" in db_url:
        parts = db_url.split("@")
        if ":" in parts[0]:
            # Mask password
            scheme_user = parts[0].split(":")[0]
            masked_url = f"{scheme_user}:****@{parts[1]}"
    logger.info("Initializing PgVectorStore", db_url=masked_url)

    vector_store = PgVectorStore(
        db_url=db_url,
        collection_prefix=settings.supabase_collection_prefix,
        embedding_dim=settings.embedding_dimensions,
    )
    await vector_store.connect()

    # Initialize Retriever
    # Initialize Embedding Client
    from src.embedder import EmbedderConfig, EmbeddingClient

    embedder_config = EmbedderConfig(
        provider=settings.default_llm_provider,
        model=settings.embedding_model,
        api_key=settings.gemini_api_key
        if settings.default_llm_provider == "gemini"
        else (settings.openai_api_key if settings.default_llm_provider == "openai" else ""),
        dimension=settings.embedding_dimensions,
    )
    embedding_client = EmbeddingClient(config=embedder_config)
    # Expose on app.state so HTTP routes (POST /mcp/v1/embeddings) can use it.
    # Previously this was scoped to lifespan() only; the route shielva-presence
    # has been calling (TrailFollower.prepare() → POST /mcp/v1/embeddings) was
    # never implemented because nothing exposed the client. TrailFollower
    # silently received [] back. Wiring the client to app.state lets the new
    # route below pull it via request.app.state.embedding_client.
    app.state.embedding_client = embedding_client

    rag_client = HybridRetriever(
        vector_store=vector_store,
        embedding_client=embedding_client,
        vector_weight=0.7,
        bm25_weight=0.3,
    )

    # Bot Registry (with MongoDB)
    bot_registry = BotRegistry(mongodb_client=mongo_client)

    # Register codegen intelligence tools (used by fix-agent endpoint)
    from src.tools.codegen_tools import register_codegen_tools

    register_codegen_tools(tool_registry)

    # Register meeting TMS tools (used by post-meeting transcript extraction)
    from src.tools.meeting_tools import register_meeting_tools

    register_meeting_tools(tool_registry)

    # Inject dependencies into tool_registry
    tool_registry.set_rag_client(rag_client)
    tool_registry.set_bot_registry(bot_registry)

    # KB Registry
    kb_registry = KBRegistry()

    # P1: Query cache (Redis with in-memory fallback)
    query_cache = None
    try:
        from cache.query_cache import InMemoryCache, RedisCache

        try:
            query_cache = RedisCache(
                redis_url=settings.redis_url,
                ttl=settings.cache_ttl_seconds,
            )
            logger.info("RAG query cache: Redis", ttl=settings.cache_ttl_seconds)
        except Exception as redis_err:
            logger.warning(
                "Redis unavailable, falling back to in-memory cache",
                error=str(redis_err),
            )
            query_cache = InMemoryCache(ttl=settings.cache_ttl_seconds)
    except ImportError:
        logger.warning("query_cache module not importable; RAG caching disabled")

    # P3: KB router (routes queries to relevant KB subset)
    kb_router = None
    if settings.kb_routing_enabled:
        try:
            from src.routing.kb_router import KBRouter

            kb_router = KBRouter(
                embedding_client=embedding_client,
                top_kbs=settings.kb_routing_top_kbs,
            )
            logger.info("KB routing enabled", top_kbs=settings.kb_routing_top_kbs)
        except Exception as router_err:
            logger.warning("KB router init failed", error=str(router_err))

    # Context assembler needs RAG client
    context_assembler = ContextAssembler(
        rag_client=rag_client,
        bot_registry=bot_registry,
        session_store=None,  # TODO: Initialize Redis session store
        prompt_engine=None,  # TODO: Initialize prompt engine
        query_cache=query_cache,
        kb_router=kb_router,
    )

    # Discovery Registration
    import asyncio

    from shared.discovery_client import DiscoveryClient

    gateway_url = os.getenv("GATEWAY_URL", "https://localhost:8000")
    api_port = int(os.getenv("MCP_PORT", "8004"))

    # Match registration scheme to the actual listen scheme — if CERT_FILE+KEY_FILE
    # are set the server serves HTTPS, so registering as plain http would point
    # callers at a non-existent listener. Same logic CMS + tms-core already use.
    cert_file = os.getenv("CERT_FILE")
    key_file = os.getenv("KEY_FILE")
    use_ssl = cert_file and key_file and os.path.exists(cert_file) and os.path.exists(key_file)
    scheme = "https" if use_ssl else "http"

    app.state.discovery = DiscoveryClient(
        service_name="mcp",
        service_port=api_port,
        gateway_url=gateway_url,
        scheme=scheme,
    )
    app.state.discovery_task = asyncio.create_task(app.state.discovery.start())
    logger.info("MCP Server registered with Discovery", gateway=gateway_url, scheme=scheme)

    # Provisioning control-plane handler (KB / bot / test-bot). Query
    # traffic is served by the DDD HandleQueryUseCase, wired below.
    message_handler = MessageHandler(kb_registry=kb_registry)

    # Surface legacy registries on app.state — still consumed by
    # codegen fix-agent + JSON-RPC dispatcher's resources/* fallback
    # paths until the slice 4c migration lands.
    app.state.kb_registry = kb_registry
    app.state.bot_registry = bot_registry
    app.state.message_handler = message_handler

    # All new-layer ports + application services are wired in one
    # place: composition.wire_use_cases. main.py owns the legacy
    # infra construction; composition owns the DDD/hexagonal graph
    # we build on top.
    from src.composition import wire_use_cases

    wire_use_cases(
        app,
        tool_registry=tool_registry,
        kb_registry=kb_registry,
        bot_registry=bot_registry,
        policy_engine=policy_engine,
        context_assembler=context_assembler,
        llm_router=llm_router,
        embedding_client=embedding_client,
        vector_store=vector_store,
        rag_client=rag_client,
    )

    logger.info("MCP Server initialized successfully")

    yield

    # Discovery Cleanup
    if hasattr(app.state, "discovery"):
        await app.state.discovery.stop()

    # Cleanup
    logger.info("Shutting down MCP Server")

    # Close MongoDB
    if mongo_client:
        mongo_client.close()
        logger.info("Closed MongoDB connection")

    # Close vector store connection
    try:
        # We need to access the vector_store from context_assembler or keep a ref
        # context_assembler.rag_client.vector_store
        if context_assembler and context_assembler.rag_client:
            await context_assembler.rag_client.vector_store.close()
            logger.info("Closed vector store connection")
    except Exception as e:
        logger.error("Error closing vector store", error=str(e))

    # Close policy engine
    try:
        if policy_engine:
            await policy_engine.close()
            logger.info("Closed policy engine connection")
    except Exception as e:
        logger.error("Error closing policy engine", error=str(e))


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="The AI Operating System for Shielva ARC",
    lifespan=lifespan,
)

# Exception handlers — must be installed before middleware and routes.
from src.core.error_handlers import install_exception_handlers

install_exception_handlers(app)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    cid = request.headers.get("X-Correlation-Id") or uuid.uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=cid)
    request.state.request_id = cid
    response = await call_next(request)
    response.headers["X-Correlation-Id"] = cid
    return response


# SOP observability: traces + /sop-metrics + optional self-registration.
# Must run before other middleware so the metrics middleware sees the
# pre-auth view of every request.
from shielva_common.sop_sdk import setup_sop

setup_sop(app, service_name="shielva-mcp")

# All REST routes are owned by interface/http/. URL paths are
# unchanged so integration-builder + ingestion-worker + presence
# clients keep working without config updates. main.py only knows
# the package; each module owns one logical surface.
from src.interface.http import (
    admin_router,
    codegen_router,
    connectors_router,
    embeddings_router,
    health_router,
    provision_router,
    query_router,
    tools_router,
)
from src.interface.http import llm_router as llm_api_router

app.include_router(health_router)
app.include_router(codegen_router)
app.include_router(query_router)
app.include_router(embeddings_router)
app.include_router(provision_router)
app.include_router(tools_router)
app.include_router(connectors_router)
app.include_router(admin_router)
app.include_router(llm_api_router)

# Industry-grade Model Context Protocol (spec 2024-11-05, Streamable
# HTTP 2025-03-26). Routes POST/DELETE /mcp through a JSON-RPC 2.0
# façade wired via the DDD/hexagonal composition root. Slice 1 wires
# the chat bounded context end-to-end; tools/resources/prompts still
# bridge to the legacy registries until later slices land. The
# internal REST API (/mcp/v1/...) keeps its existing shape — every
# other consumer is unaffected.
from src.composition import build_mcp_jsonrpc_router

app.include_router(build_mcp_jsonrpc_router())

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=json.loads(
        os.getenv(
            "CORS_ORIGINS",
            '["https://localhost:3010","https://localhost:3001","http://localhost:3010","http://localhost:3000","https://localhost:3000","https://localhost:3005","https://127.0.0.1:3010","http://127.0.0.1:3000"]',
        )
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main() -> None:
    """uvicorn entry point — invoked by ``python -m src.main``."""
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )


if __name__ == "__main__":
    main()
