"""
Vector Store Data Models
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorDocument:
    """Document to be stored in vector DB"""

    id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result from vector search"""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
