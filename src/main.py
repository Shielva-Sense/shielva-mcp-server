"""
Shielva MCP Server - Main Application
The AI Operating System for Shielva ARC
"""
from fastapi.responses import Response
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from typing import Dict, Any
import structlog
import uvicorn
import json
import os

from config.settings import get_settings
from src.protocol.models import (
    MCPQueryRequest, MCPQueryResponse,
    ProvisionKBRequest, ProvisionKBResponse,
    ProvisionBotRequest, ProvisionBotResponse,
    TestBotRequest, TestBotResponse,
    TenantContext,
    ToolExecutionRequest, ToolExecutionResponse,
    ConnectorSyncRequest, ConnectorSyncResponse
)
from src.protocol.message_handler import MessageHandler
from src.context.assembler import ContextAssembler
from src.routing.llm_router import LLMRouter
from src.registry.bot_registry import BotRegistry
from src.registry.kb_registry import KBRegistry
from src.registry.tool_registry import ToolRegistry, create_registry_with_defaults

# Configure logging
import logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout
)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

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
        fallback_models=settings.litellm_fallback_models
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
        policy_path="/v1/data/shielva"
    )
    
    # RAG Engine - Supabase Vector
    from src.rag_engine.vectorstore import SupabaseVectorStore
    from src.rag_engine.retriever import HybridRetriever
    from src.rag_engine.vectorstore.supabase_store import SupabaseVectorStore
    
    # We need to make sure src.rag_engine imports resolve correctly.
    # Assuming 'rag-engine/src' is in PYTHONPATH or symlinked?
    # Given project structure, we might need to adjust imports if not installed as package.
    # But let's assume standard import path based on structure.
    # Actually, the rag-engine seems to be separate.
    # I'll use local imports if needed or assume installed.
    # For now, I'll assume the files are importable.
    
    # Initialize MongoDB
    from motor.motor_asyncio import AsyncIOMotorClient
    try:
        import certifi as _c; _tls = {"tlsCAFile": _c.where()} if settings.mongodb_url.startswith("mongodb+srv") else {}
    except ImportError:
        _tls = {}
    mongo_client = AsyncIOMotorClient(settings.mongodb_url, **_tls)
    
    # Initialize Vector Store
    # Use SUPABASE_DB_URL if available, else construct/warn
    db_url = settings.supabase_db_url
    if not db_url and settings.supabase_url:
        # Warn user or try to construct? 
        # For now we rely on db_url being set in server.sh
        logger.warning("SUPABASE_DB_URL not set. Vector store may fail to connect.")
        db_url = "postgresql://postgres:postgres@localhost:54322/postgres" # Fallback/Invalid
    
    # DEBUG: Print DB URL (mask password)
    masked_url = db_url
    if "@" in db_url:
        parts = db_url.split("@")
        if ":" in parts[0]:
            # Mask password
            scheme_user = parts[0].split(":")[0]
            masked_url = f"{scheme_user}:****@{parts[1]}"
    logger.info("Initializing SupabaseVectorStore", db_url=masked_url)

    vector_store = SupabaseVectorStore(
        db_url=db_url,
        collection_prefix=settings.supabase_collection_prefix,
        embedding_dim=settings.embedding_dimensions
    )
    await vector_store.connect()
    
    # Initialize Retriever
    # Initialize Embedding Client
    from src.embedder import EmbeddingClient, EmbedderConfig
    
    embedder_config = EmbedderConfig(
        provider=settings.default_llm_provider,
        model=settings.embedding_model,
        api_key=settings.gemini_api_key if settings.default_llm_provider == "gemini" else (settings.openai_api_key if settings.default_llm_provider == "openai" else ""),
        dimension=settings.embedding_dimensions
    )
    embedding_client = EmbeddingClient(config=embedder_config)

    rag_client = HybridRetriever(
        vector_store=vector_store,
        embedding_client=embedding_client, 
        vector_weight=0.7,
        bm25_weight=0.3
    )

    # Bot Registry (with MongoDB)
    bot_registry = BotRegistry(mongodb_client=mongo_client)
    
    # Register codegen intelligence tools (used by fix-agent endpoint)
    from src.tools.codegen_tools import register_codegen_tools
    register_codegen_tools(tool_registry)

    # Inject dependencies into tool_registry
    tool_registry.set_rag_client(rag_client)
    tool_registry.set_bot_registry(bot_registry)
    
    # KB Registry
    kb_registry = KBRegistry()

    # P1: Query cache (Redis with in-memory fallback)
    query_cache = None
    try:
        from cache.query_cache import RedisCache, InMemoryCache
        try:
            query_cache = RedisCache(
                redis_url=settings.redis_url,
                ttl=settings.cache_ttl_seconds,
            )
            logger.info("RAG query cache: Redis", ttl=settings.cache_ttl_seconds)
        except Exception as redis_err:
            logger.warning("Redis unavailable, falling back to in-memory cache", error=str(redis_err))
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
    gateway_url = os.getenv("GATEWAY_URL", "http://localhost:8000")
    api_port = int(os.getenv("MCP_PORT", 8004))
    
    app.state.discovery = DiscoveryClient(
        service_name="mcp",
        service_port=api_port,
        gateway_url=gateway_url
    )
    asyncio.create_task(app.state.discovery.start())
    logger.info("MCP Server registered with Discovery", gateway=gateway_url)

    # Message handler orchestrates everything
    message_handler = MessageHandler(
        context_assembler=context_assembler,
        tool_registry=tool_registry,
        kb_registry=kb_registry,
        llm_router=llm_router,
        policy_engine=policy_engine
    )
    
    logger.info("MCP Server initialized successfully")

    # Start RAG ingestion scheduler (daily auto-reindex)
    start_scheduler()

    yield

    # Stop scheduler
    stop_scheduler()

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
        # We need to access policy_engine from message_handler
        # message_handler.policy_engine
        if message_handler and message_handler.policy_engine:
            await message_handler.policy_engine.close()
            logger.info("Closed policy engine connection")
    except Exception as e:
        logger.error("Error closing policy engine", error=str(e))


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="The AI Operating System for Shielva ARC",
    lifespan=lifespan
)

# Internal codegen routes (used by integration-builder when LLM_MODE=mcp)
from src.api.codegen_routes import codegen_router  # noqa: E402
app.include_router(codegen_router)

# RAG ingestion management (start/cancel/status/stats/schedule for security-fix-collector)
from src.api.ingest_routes import router as ingest_router, start_scheduler, stop_scheduler  # noqa: E402
app.include_router(ingest_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=json.loads(os.getenv("CORS_ORIGINS", '["https://localhost:3010","https://localhost:3001","http://localhost:3010","http://localhost:3000","https://localhost:3000","https://localhost:3005","https://127.0.0.1:3010","http://127.0.0.1:3000"]')),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Dependencies =====

async def get_tenant_context_async(request: Request) -> TenantContext:
    """Extract tenant context from request headers and fetch permissions from MongoDB."""
    tenant_id = request.headers.get("X-Tenant-ID")
    user_id = request.headers.get("X-User-ID")
    user_email = request.headers.get("X-User-Email")
    role = request.headers.get("X-User-Role", "Customer_Basic")
    
    if not tenant_id:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Tenant-ID header"
        )
    
    # Fetch custom permissions from MongoDB
    permissions = []
    if user_email:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            mongodb_url = os.getenv("MONGODB_URL", "mongodb+srv://shielvaadmin:shielvaadmin123@mastershielva.8rbs44q.mongodb.net/?appName=MasterShielva")
            try:
                import certifi as _c2; _tls2 = {"tlsCAFile": _c2.where()} if mongodb_url.startswith("mongodb+srv") else {}
            except ImportError:
                _tls2 = {}
            client = AsyncIOMotorClient(mongodb_url, **_tls2)
            db = client["CustomerProfile"]
            collection = db["llm_user_configuration"]
            
            user_config = await collection.find_one({"user_email": user_email})
            if user_config and "access_permissions" in user_config:
                permissions = user_config["access_permissions"]
                logger.info(
                    "Loaded custom permissions from MongoDB",
                    user_email=user_email,
                    permissions=permissions
                )
            
            client.close()
        except Exception as e:
            logger.warning(
                "Failed to fetch custom permissions from MongoDB",
                user_email=user_email,
                error=str(e)
            )
    
    return TenantContext(
        tenant_id=tenant_id,
        user_id=user_id or "unknown",
        user_email=user_email or "unknown@example.com",
        role=role,
        permissions=permissions
    )

def get_tenant_context(request: Request) -> TenantContext:
    """Synchronous wrapper for get_tenant_context_async (for backwards compatibility)."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(get_tenant_context_async(request))



# ===== Health Endpoints =====

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "mcp-server",
        "version": settings.app_version
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check with component status."""
    return {
        "status": "ready",
        "components": {
            "tool_registry": tool_registry is not None,
            "llm_router": llm_router is not None,
            "context_assembler": context_assembler is not None,
            "message_handler": message_handler is not None
        }
    }


# ===== Query Endpoints =====

@app.post(f"{settings.api_prefix}/query", response_model=MCPQueryResponse)
async def process_query(
    request: MCPQueryRequest,
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """
    Process a query through the MCP pipeline.
    
    This is the main entry point for AI queries.
    """
    logger.info(
        "Query request received",
        tenant_id=tenant_context.tenant_id,
        bot_id=request.bot_id
    )
    
    try:
        response = await message_handler.handle_query(
            request=request,
            tenant_context=tenant_context
        )
        return response
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error("Query processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.api_prefix}/query/stream")
async def process_query_stream(
    request: MCPQueryRequest,
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """Process query with streaming response."""
    request.stream = True
    
    async def generate():
        try:
            response = await message_handler.handle_query(
                request=request,
                tenant_context=tenant_context
            )
            yield response.answer
        except Exception as e:
            yield f"Error: {str(e)}"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


# ===== Provisioning Endpoints =====

@app.post(f"{settings.api_prefix}/provision/kb", response_model=ProvisionKBResponse)
async def provision_kb(
    request: ProvisionKBRequest,
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """
    Provision a new Knowledge Base.
    
    Triggers the KB installation workflow:
    1. Register KB
    2. Generate hello_id
    3. Trigger connector sync
    4. Start ingestion
    """
    logger.info(
        "KB provision request",
        tenant_id=tenant_context.tenant_id,
        kb_name=request.kb_config.name
    )
    
    try:
        response = await message_handler.handle_provision_kb(
            request=request,
            tenant_context=tenant_context
        )
        return response
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error("KB provisioning failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.api_prefix}/provision/bot", response_model=ProvisionBotResponse)
async def provision_bot(
    request: ProvisionBotRequest,
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """Provision a new bot."""
    logger.info(
        "Bot provision request",
        tenant_id=tenant_context.tenant_id,
        bot_name=request.name
    )
    
    try:
        response = await message_handler.handle_provision_bot(
            request=request,
            tenant_context=tenant_context
        )
        return response
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== Testing Endpoints =====

@app.post(f"{settings.api_prefix}/test", response_model=TestBotResponse)
async def test_bot(
    request: TestBotRequest,
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """
    Run automated tests against a bot.
    
    Used to validate bot before activation.
    """
    logger.info(
        "Bot test request",
        tenant_id=tenant_context.tenant_id,
        bot_id=request.bot_id
    )
    
    try:
        response = await message_handler.handle_test_bot(
            request=request,
            tenant_context=tenant_context
        )
        return response
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{settings.api_prefix}/activate")
async def activate_bot(
    bot_id: str,
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """Activate a bot after successful testing."""
    logger.info(
        "Bot activation request",
        tenant_id=tenant_context.tenant_id,
        bot_id=bot_id
    )
    
    # TODO: Implement activation logic
    return {"status": "activated", "bot_id": bot_id}


# ===== Tool Endpoints =====

@app.get(f"{settings.api_prefix}/tools")
async def list_tools(
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """List available tools."""
    tools = tool_registry.get_all_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "parameters": [p.dict() for p in t.parameters]
            }
            for t in tools
        ]
    }


@app.post(f"{settings.api_prefix}/tools/{{tool_name}}/execute", response_model=ToolExecutionResponse)
async def execute_tool(
    tool_name: str,
    request: ToolExecutionRequest,
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """Execute a tool directly."""
    request.tool_name = tool_name
    
    response = await tool_registry.execute_tool(
        request=request,
        tenant_context=tenant_context
    )
    
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error)
    
    return response


# ===== Connector Endpoints =====

@app.post(f"{settings.api_prefix}/connectors/sync", response_model=ConnectorSyncResponse)
async def trigger_connector_sync(
    request: ConnectorSyncRequest,
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """Trigger a connector sync."""
    logger.info(
        "Connector sync triggered",
        tenant_id=tenant_context.tenant_id,
        connector_id=request.connector_id
    )
    
    # TODO: Call connector gateway
    return ConnectorSyncResponse(
        job_id="sync-job-123",
        status="syncing",
        documents_found=0,
        message="Sync initiated"
    )


@app.get(f"{settings.api_prefix}/connectors/{{connector_id}}/status")
async def get_connector_status(
    connector_id: str,
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """Get connector status."""
    # TODO: Get from connector registry
    return {
        "connector_id": connector_id,
        "health": "healthy",
        "last_sync": None,
        "documents_indexed": 0
    }


# ===== Admin Endpoints =====

@app.get(f"{settings.api_prefix}/admin/stats")
async def get_stats(
    tenant_context: TenantContext = Depends(get_tenant_context)
):
    """Get MCP server statistics."""
    return {
        "tenant_id": tenant_context.tenant_id,
        "queries_processed": 0,  # TODO: Track
        "tools_available": len(tool_registry.get_all_tools()) if tool_registry else 0,
        "active_sessions": 0  # TODO: Track
    }


# ===== Run Server =====

def main():
    """Run the MCP server."""
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )




@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Expose Prometheus metrics for scraping."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        from fastapi.responses import Response
        return Response(status_code=404, content=b"prometheus_client not installed")

if __name__ == "__main__":
    main()
