"""
MCP Server Configuration Settings
---------------------------------
SealedSettings — secrets MUST come from envelope-decrypted env, file-mount,
or kwargs. Plaintext defaults are rejected at boot.

See shielva_common/config/sealed.py for the full contract.
"""
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, SecretStr

from shielva_common.config.sealed import SealedSettings, sealed_field


class MCPSettings(SealedSettings):
    """MCP Server configuration.

    Non-secret values load from env normally. Secret values (DB URI with
    password, API keys, JWT signing key) must be sealed — see
    docs/compliance/SECRETS_AT_REST.md for the canonical pattern.
    """

    # ── Application (non-secret) ─────────────────────────────────────────
    service_name: str = Field("shielva-mcp", validation_alias="SERVICE_NAME")
    app_name: str = "Shielva MCP Server"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = Field("development", validation_alias="ENVIRONMENT")

    # ── API (non-secret) ─────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = Field(8004, validation_alias="MCP_PORT")
    api_prefix: str = "/mcp/v1"

    # ── MongoDB (SECRET — full URI carries password) ─────────────────────
    # MUST come from envelope.bootstrap() or MONGODB_URL_FILE. No default
    # creds. Older deploys used mongodb+srv://... with embedded password;
    # that pattern is forbidden — point this at the sealed config blob.
    mongodb_url: SecretStr = sealed_field(
        ...,
        env="MONGODB_URL",
        file_env="MONGODB_URL_FILE",
        description="MongoDB connection URI (sealed)",
    )
    mongodb_db_name: str = "CustomerProfile"

    # ── PostgreSQL (SECRET — connection URI carries password) ────────────
    postgres_url: SecretStr = sealed_field(
        SecretStr(""),
        env="POSTGRES_URL",
        file_env="POSTGRES_URL_FILE",
        description="Postgres connection URI for knowledge-manager persistence",
    )

    # ── Supabase Vector DB (SECRET) ──────────────────────────────────────
    supabase_url: Optional[str] = None
    supabase_key: SecretStr = sealed_field(
        SecretStr(""),
        env="SUPABASE_KEY",
        file_env="SUPABASE_KEY_FILE",
    )
    supabase_db_url: SecretStr = sealed_field(
        SecretStr(""),
        env="SUPABASE_DB_URL",
        file_env="SUPABASE_DB_URL_FILE",
    )
    supabase_collection_prefix: str = "shielva_kb_"

    # ── Redis (URI may carry password) ───────────────────────────────────
    redis_url: SecretStr = sealed_field(
        SecretStr(""),
        env="REDIS_URL",
        file_env="REDIS_URL_FILE",
    )
    redis_db: int = 0
    cache_ttl_seconds: int = 900  # 15 minutes

    # ── LLM Configuration (SECRET — API keys) ────────────────────────────
    default_llm_provider: str = "gemini"
    openai_api_key: SecretStr = sealed_field(
        SecretStr(""), env="OPENAI_API_KEY", file_env="OPENAI_API_KEY_FILE",
    )
    anthropic_api_key: SecretStr = sealed_field(
        SecretStr(""), env="ANTHROPIC_API_KEY", file_env="ANTHROPIC_API_KEY_FILE",
    )
    azure_openai_api_key: SecretStr = sealed_field(
        SecretStr(""), env="AZURE_OPENAI_API_KEY", file_env="AZURE_OPENAI_API_KEY_FILE",
    )
    azure_openai_endpoint: Optional[str] = None
    gemini_api_key: SecretStr = sealed_field(
        SecretStr(""), env="GEMINI_API_KEY", file_env="GEMINI_API_KEY_FILE",
    )

    # ── LiteLLM Config (non-secret) ──────────────────────────────────────
    litellm_model: str = "gemini/gemini-2.5-flash-lite"
    litellm_fallback_models: List[str] = ["gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro"]
    max_tokens: int = 65536
    temperature: float = 0.1

    # ── Per-tenant LLM routing (platform-backend internal resolver) ──────
    # When set, MCP asks shielva-platform for the calling tenant's active
    # provider/model + decrypted BYOK key (GET /internal/llm/resolve) and routes
    # that tenant's inference accordingly. Empty url/token → feature off (every
    # tenant uses litellm_model above, today's behavior). Result cached per
    # tenant for tenant_llm_cache_ttl seconds. The token MUST match
    # shielva-platform's PLATFORM_INTERNAL_TOKEN.
    platform_internal_url: str = Field("", validation_alias="PLATFORM_INTERNAL_URL")
    platform_internal_token: SecretStr = sealed_field(
        SecretStr(""), env="PLATFORM_INTERNAL_TOKEN", file_env="PLATFORM_INTERNAL_TOKEN_FILE",
    )
    tenant_llm_cache_ttl: int = 60
    # Verify TLS on the resolver call. DEV: localhost uses self-signed certs →
    # default False. PROD: set PLATFORM_INTERNAL_VERIFY_TLS=true.
    tenant_llm_verify_tls: bool = Field(False, validation_alias="PLATFORM_INTERNAL_VERIFY_TLS")

    # ── Embedding (non-secret) ───────────────────────────────────────────
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dimensions: int = 768

    # ── RAG Settings (non-secret) ────────────────────────────────────────
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ── KB Routing (non-secret) ──────────────────────────────────────────
    kb_routing_enabled: bool = True
    kb_routing_top_kbs: int = 2

    # ── Connector Gateway (non-secret URL) ───────────────────────────────
    connector_gateway_url: str = "https://localhost:8003"
    connector_timeout_seconds: int = 30

    # ── Subscription entitlement gate (non-secret) ───────────────────────
    # The LLM broker enforces subscription entitlements at the chokepoint:
    # every LLM-serving endpoint asks shielva-billing whether the tenant is
    # entitled before serving. billing owns the managed-vs-self_hosted logic
    # (a provisioned subscription fans plan entitlements → active; an
    # unprovisioned self-hosted tenant has none), so MCP needs no
    # deployment_type of its own. DISABLED by default — a new gate on the hot
    # path is opt-in per deployment, enabled once billing is populated.
    entitlement_enabled: bool = False
    billing_url: str = "https://localhost:8700"
    entitlement_llm_feature_key: str = "llm"
    entitlement_cache_ttl_seconds: int = 60
    # Billing unreachable → False (default) FAIL OPEN (serve + warn, so a
    # billing blip never takes down all LLM traffic); True → FAIL CLOSED (deny).
    entitlement_fail_closed: bool = False

    # ── Security (SECRET — JWT signing material) ─────────────────────────
    jwt_secret_key: SecretStr = sealed_field(
        ...,
        env="JWT_SECRET_KEY",
        file_env="JWT_SECRET_KEY_FILE",
        description="HMAC signing key for service-issued JWTs",
    )
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60

    # ── Gateway-stamped principal HMAC (SECRET) ──────────────────────────
    # Used by shielva_common.auth.configure_principal_verification.
    gateway_principal_hmac_key: SecretStr = sealed_field(
        SecretStr(""),
        env="GATEWAY_PRINCIPAL_HMAC_KEY",
        file_env="GATEWAY_PRINCIPAL_HMAC_KEY_FILE",
    )

    # ── Audit HMAC (SECRET) ──────────────────────────────────────────────
    audit_hmac_secret: SecretStr = sealed_field(
        ...,
        env="AUDIT_HMAC_SECRET",
        file_env="AUDIT_HMAC_SECRET_FILE",
        description="HMAC-SHA256 key for the per-service audit chain",
    )

    # ── Vault sidecar (non-secret URL) ───────────────────────────────────
    vault_url: Optional[str] = Field(None, validation_alias="VAULT_SIDECAR_URL")
    vault_token: SecretStr = sealed_field(
        SecretStr(""), env="VAULT_TOKEN", file_env="VAULT_TOKEN_FILE",
    )

    # ── Policy Engine (non-secret URL) ───────────────────────────────────
    opa_url: Optional[str] = None

    # ── Observability (SECRET keys) ──────────────────────────────────────
    otlp_endpoint: Optional[str] = None
    langsmith_api_key: SecretStr = sealed_field(
        SecretStr(""), env="LANGSMITH_API_KEY", file_env="LANGSMITH_API_KEY_FILE",
    )
    langsmith_project: str = "shielva-mcp"

    # ── Observability (non-secret) ───────────────────────────────────────
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")

    # ── Rate Limiting (non-secret) ───────────────────────────────────────
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60

    # ── Security Fix RAG (non-secret) ────────────────────────────────────
    security_fix_rag_enabled: bool = True
    security_fix_vector_table: str = "security_fix_entries"
    security_fix_top_k: int = 5
    security_fix_min_similarity: float = 0.45


@lru_cache()
def get_settings() -> MCPSettings:
    """Get cached settings instance.

    On first call, SealedSettings refuses any plaintext default and
    surfaces the missing field via an EnvelopeSealingError. Callers
    should run shielva_common.envelope.bootstrap() first (or set
    SHIELVA_SEALED_PERMISSIVE=1 for dev migrations).
    """
    return MCPSettings()
