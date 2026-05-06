"""
MCP Server Configuration Settings
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List
from functools import lru_cache


class MCPSettings(BaseSettings):
    """MCP Server configuration"""
    
    # Application
    app_name: str = "Shielva MCP Server"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "development"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = Field(8004, validation_alias="MCP_PORT")
    api_prefix: str = "/mcp/v1"
    
    # MongoDB
    mongodb_url: str = "mongodb+srv://shielvaadmin:shielvaadmin123@mastershielva.8rbs44q.mongodb.net/?appName=MasterShielva"
    mongodb_db_name: str = "CustomerProfile"
    
    # PostgreSQL (for structured data)
    postgres_url: str = "postgresql+asyncpg://localhost:5432/shielva_mcp"
    
    # Supabase Vector DB (pgvector)
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    supabase_db_url: Optional[str] = None # Connection string for vecs
    supabase_collection_prefix: str = "shielva_kb_"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    cache_ttl_seconds: int = 900  # 15 minutes
    
    # LLM Configuration
    default_llm_provider: str = "gemini"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = Field(None, validation_alias="ANTHROPIC_API_KEY")
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    gemini_api_key: Optional[str] = Field(None, validation_alias="GEMINI_API_KEY")
    
    # LiteLLM Config
    litellm_model: str = "gemini/gemini-2.5-flash-lite"
    litellm_fallback_models: List[str] = ["gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro"]
    max_tokens: int = 65536
    temperature: float = 0.1
    
    # Embedding
    embedding_model: str = "models/gemini-embedding-001" # Gemini embedding model
    embedding_dimensions: int = 768  # truncated via outputDimensionality to match DB
    
    # RAG Settings
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50

    # KB Routing (Vectorless RAG — query routing to relevant KB subset)
    kb_routing_enabled: bool = True
    kb_routing_top_kbs: int = 2
    
    # Connector Gateway
    connector_gateway_url: str = "http://localhost:8003"
    connector_timeout_seconds: int = 30
    
    # Security
    jwt_secret_key: str = "your-jwt-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    
    # Vault (for secrets)
    vault_url: Optional[str] = None
    vault_token: Optional[str] = None
    
    # Policy Engine
    opa_url: Optional[str] = None  # OPA for policy decisions
    
    # Observability
    otlp_endpoint: Optional[str] = None
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "shielva-mcp"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # Security Fix RAG — vulnerability fix suggestion engine
    security_fix_rag_enabled: bool = True
    security_fix_vector_table: str = "security_fix_entries"
    security_fix_top_k: int = 5
    security_fix_min_similarity: float = 0.45
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"   # silently ignore unknown env vars (e.g. DATABASE_URL)


@lru_cache()
def get_settings() -> MCPSettings:
    """Get cached settings instance"""
    return MCPSettings()
