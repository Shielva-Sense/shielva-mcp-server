"""Persistence adapters — implement repository ports defined by the domain.

Slice 1 ships the in-memory chat session repo.
Slice 3 adds the legacy KB repository adapter wrapping the existing
in-memory ``KBRegistry``. A Mongo-backed sibling adapter will land
in slice 4 alongside the bot context's persistence move.
"""
from .in_memory_chat_session_repository import InMemoryChatSessionRepository
from .legacy_bot_repository_adapter import LegacyBotRepositoryAdapter
from .legacy_kb_repository_adapter  import LegacyKBRepositoryAdapter
from .mongo_user_config_repository  import (
    MongoUserConfigRepository,
    get_user_config_repo,
)

__all__ = [
    "InMemoryChatSessionRepository",
    "LegacyBotRepositoryAdapter",
    "LegacyKBRepositoryAdapter",
    "MongoUserConfigRepository",
    "get_user_config_repo",
]
