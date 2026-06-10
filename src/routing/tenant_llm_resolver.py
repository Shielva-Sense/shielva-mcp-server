"""Per-tenant LLM resolution.

Asks shielva-platform's internal resolver (GET /internal/llm/resolve) for the
calling tenant's active provider/model + decrypted BYOK key, maps it to a
LiteLLM model string (+ api_base for OpenAI-compatible providers), and caches
the result per tenant for a short TTL.

Design notes:
  • shielva-platform is the SOLE holder of the at-rest crypto + master key — MCP
    never decrypts. The resolver hands back a plaintext key over an internal,
    service-token-authenticated, non-gateway channel.
  • Fail-OPEN to the platform default: any miss (no config / managed tier /
    network error / unmapped provider) returns None, and the caller keeps using
    settings.litellm_model. This feature can only ADD per-tenant routing, never
    break inference for tenants without a config.
  • The plaintext key lives only in the in-process cache for tenant_llm_cache_ttl
    seconds and is never logged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import httpx
import structlog

from config.settings import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# provider key (as stored by shielva-platform) → (litellm model template, api_base)
# DeepSeek + Kimi/Moonshot are OpenAI-compatible → openai/ prefix + custom base.
_PROVIDER_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "gemini":   ("gemini/{model}",    None),
    "claude":   ("anthropic/{model}", None),
    "openai":   ("openai/{model}",    None),
    "groq":     ("groq/{model}",      None),
    "mistral":  ("mistral/{model}",   None),
    "deepseek": ("openai/{model}",    "https://api.deepseek.com/v1"),
    "kimi":     ("openai/{model}",    "https://api.moonshot.cn/v1"),
}


def format_model_for_provider(provider: str, model: str) -> str:
    """Apply a provider's LiteLLM template to a raw model id, e.g.
    ('gemini', 'gemini-2.5-pro') → 'gemini/gemini-2.5-pro'. Used for per-bot model
    overrides (the bot stores a bare model id; the provider+key come from the
    tenant config). Already-prefixed ('x/y') or unknown-provider → returned as-is."""
    if not model or "/" in model:
        return model
    spec = _PROVIDER_MAP.get((provider or "").lower())
    return spec[0].format(model=model) if spec else model


@dataclass
class ResolvedLLM:
    """A tenant's resolved routing target."""
    model: str                      # LiteLLM model string, e.g. "openai/gpt-4o"
    api_key: Optional[str]          # decrypted BYOK key (may be None)
    api_base: Optional[str] = None  # for OpenAI-compatible providers
    provider: str = ""              # raw provider key (e.g. "gemini") — used to format per-bot overrides


class TenantLLMResolver:
    """Caches per-tenant LLM routing resolved from shielva-platform."""

    def __init__(self) -> None:
        # tenant_id → (fetched_at_monotonic, resolved-or-None)
        self._cache: Dict[str, Tuple[float, Optional[ResolvedLLM]]] = {}
        self._ttl = max(0, settings.tenant_llm_cache_ttl)
        self._url = settings.platform_internal_url.rstrip("/")
        self._token = settings.platform_internal_token.get_secret_value()
        self._verify = settings.tenant_llm_verify_tls
        # One reused client (keep-alive pool) instead of a fresh TLS handshake on
        # every cache miss — this sits on the critical path before the LLM call.
        self._client: Optional[httpx.AsyncClient] = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=3.0, verify=self._verify)
        return self._client

    @property
    def enabled(self) -> bool:
        return bool(self._url and self._token)

    async def resolve(self, tenant_id: Optional[str]) -> Optional[ResolvedLLM]:
        """Return the tenant's routing target, or None to use the default."""
        if not tenant_id or not self.enabled:
            return None

        now = time.monotonic()
        cached = self._cache.get(tenant_id)
        if cached is not None and (now - cached[0]) < self._ttl:
            return cached[1]

        resolved = await self._fetch(tenant_id)
        self._cache[tenant_id] = (now, resolved)
        return resolved

    def invalidate(self, tenant_id: str) -> None:
        self._cache.pop(tenant_id, None)

    async def _fetch(self, tenant_id: str) -> Optional[ResolvedLLM]:
        try:
            resp = await self._http().get(
                f"{self._url}/internal/llm/resolve",
                params={"tenant_id": tenant_id},
                headers={"X-Platform-Internal-Token": self._token},
            )
        except Exception as exc:  # network / TLS / timeout — degrade to default
            logger.warning(
                "tenant_llm_resolve_failed", tenant_id=tenant_id, error=type(exc).__name__
            )
            return None

        if resp.status_code == 204:
            return None
        if resp.status_code != 200:
            logger.warning(
                "tenant_llm_resolve_non_200", tenant_id=tenant_id, status=resp.status_code
            )
            return None

        try:
            data = resp.json()
        except Exception:
            return None
        return self._map(tenant_id, data)

    def _map(self, tenant_id: str, data: dict) -> Optional[ResolvedLLM]:
        provider = str(data.get("provider") or "").lower()
        model = str(data.get("model") or "")
        spec = _PROVIDER_MAP.get(provider)
        if spec is None or not model:
            logger.warning(
                "tenant_llm_unmapped_provider", tenant_id=tenant_id, provider=provider
            )
            return None
        template, api_base = spec
        # Do NOT log api_key.
        return ResolvedLLM(
            model=template.format(model=model),
            api_key=data.get("api_key"),
            api_base=api_base,
            provider=provider,
        )


# Module-level singleton — one cache shared across all router instances.
_resolver: Optional[TenantLLMResolver] = None


def get_tenant_llm_resolver() -> TenantLLMResolver:
    global _resolver
    if _resolver is None:
        _resolver = TenantLLMResolver()
    return _resolver
