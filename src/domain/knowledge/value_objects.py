"""Value objects for the knowledge context."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, NewType, Tuple


KBId       = NewType("KBId", str)
KBName     = NewType("KBName", str)
DocumentId = NewType("DocumentId", str)
ChunkId    = NewType("ChunkId", str)


class KBStatus(str, Enum):
    PENDING      = "pending"
    PROVISIONING = "provisioning"
    SYNCING      = "syncing"
    INDEXING     = "indexing"
    ACTIVE       = "active"
    FAILED       = "failed"
    SUSPENDED    = "suspended"


@dataclass(frozen=True, slots=True)
class Document:
    """A logical document inside a KB. The vector store may chunk it
    further; this is the addressable unit the connector ingested."""
    id:       DocumentId
    kb_id:    KBId
    title:    str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """One vectorisable slice of a Document."""
    id:          ChunkId
    document_id: DocumentId
    kb_id:       KBId
    content:     str
    metadata:    Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """Input to :class:`Retriever`. ``kb_ids`` is the candidate set;
    KB-routing strategy (which KBs to actually search) is the
    application service's call — at the port we accept the
    pre-filtered list."""
    query:   str
    kb_ids:  Tuple[KBId, ...]
    top_k:   int = 5


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A Chunk with a similarity score. The same chunk may be
    retrieved with different scores in different queries — the score
    is *part of the result*, not part of the chunk identity."""
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Source:
    """Citation surfaced to the LLM/client. This is the read-model
    view of a RetrievedChunk shaped for the MCP wire format —
    constructed by the application service from a list of
    :class:`RetrievedChunk` + KB metadata lookups."""
    kb_id:           KBId
    kb_name:         KBName
    document_id:     DocumentId
    document_title:  str
    chunk_id:        ChunkId
    content:         str       # excerpt (trimmed by the service)
    score:           float
    metadata:        Dict[str, Any] = field(default_factory=dict)
