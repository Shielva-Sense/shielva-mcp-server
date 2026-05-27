"""Knowledge-context use cases.

Slice 3 ships :class:`KnowledgeApplicationService` with:
    * list_knowledge_bases — backs MCP ``resources/list`` (via the
      JSON-RPC dispatcher's resource bridge in slice 4)
    * read_knowledge_base   — backs MCP ``resources/read``
    * retrieve_chunks       — the RAG read pipeline; composes
      EmbeddingClient + Retriever + KBRepository to produce
      ``Source`` value objects ready for LLM injection.

Provisioning (``provision_kb``) lands in slice 4 alongside the
bot context — they share an MCP REST endpoint.
"""
from .services import KnowledgeApplicationService

__all__ = ["KnowledgeApplicationService"]
