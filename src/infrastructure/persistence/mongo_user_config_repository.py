"""Mongo-backed repository for user configuration look-ups.

Encapsulates the ``CustomerProfile.llm_user_configuration`` collection
access that was previously inline in ``interface/http/_deps.py``.

The Motor client is shared via a module-level cache (same pattern as the
old inline helper) so we never open a new connection pool per request.
The ``get_user_config_repo`` FastAPI dependency function returns the
singleton instance, allowing callers to test with a stub via
``app.dependency_overrides``.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

_INSTANCE: MongoUserConfigRepository | None = None


class MongoUserConfigRepository:
    """Read-only access to ``llm_user_configuration`` for permissions look-ups."""

    _client_cache: dict = {}

    def __init__(self, mongodb_url: str) -> None:
        self._mongodb_url = mongodb_url

    def _get_client(self):  # type: ignore[return]
        """Return a cached Motor client for the configured URL."""
        from motor.motor_asyncio import AsyncIOMotorClient

        url = self._mongodb_url
        client = self.__class__._client_cache.get(url)
        if client is None:
            try:
                import certifi as _c

                tls = {"tlsCAFile": _c.where()} if url.startswith("mongodb+srv") else {}
            except ImportError:
                tls = {}
            client = AsyncIOMotorClient(url, **tls)
            self.__class__._client_cache[url] = client
        return client

    async def get_permissions(self, user_email: str) -> list[str]:
        """Return the ``access_permissions`` list for *user_email*.

        Best-effort: returns ``[]`` on any error (matches legacy behaviour).
        """
        if not user_email or user_email == "unknown@example.com":
            return []
        try:
            client = self._get_client()
            db = client["CustomerProfile"]
            user_config = await db["llm_user_configuration"].find_one({"user_email": user_email})
            if user_config and "access_permissions" in user_config:
                return list(user_config["access_permissions"])
        except Exception as exc:
            logger.warning(
                "user_config_repo.get_permissions_failed",
                user_email=user_email,
                error=str(exc),
            )
        return []


def get_user_config_repo() -> MongoUserConfigRepository:
    """FastAPI dependency — returns the process-level singleton.

    The singleton is created on first call (lazy) using the sealed
    settings URL so the settings loader has already run by then.
    Callers can override via ``app.dependency_overrides[get_user_config_repo]``
    to inject a stub during testing.
    """
    global _INSTANCE
    if _INSTANCE is None:
        from config.settings import get_settings

        settings = get_settings()
        mongodb_url = settings.mongodb_url.get_secret_value()
        if not mongodb_url:
            # No Mongo configured — return a no-op repo so the dep never
            # raises; get_permissions will return [] for every call.
            mongodb_url = ""
        _INSTANCE = MongoUserConfigRepository(mongodb_url=mongodb_url)
    return _INSTANCE
