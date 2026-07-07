"""
MCP Protocol - Core Data Models
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message roles in conversation"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCallStatus(str, Enum):
    """Status of tool execution"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class KBStatus(str, Enum):
    """Knowledge base status"""

    PENDING = "pending"
    PROVISIONING = "provisioning"
    SYNCING = "syncing"
    INDEXING = "indexing"
    TESTING = "testing"
    ACTIVE = "active"
    FAILED = "failed"
    SUSPENDED = "suspended"


class BotStatus(str, Enum):
    """Bot status"""

    DRAFT = "draft"
    TESTING = "testing"
    ACTIVE = "active"
    SUSPENDED = "suspended"


# ===== Core Protocol Messages =====


class MCPMessage(BaseModel):
    """Base MCP protocol message"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    metadata: dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TenantContext(BaseModel):
    """Tenant isolation context"""

    tenant_id: str
    user_id: str
    user_email: str
    role: str = "Customer_Basic"
    permissions: list[str] = []
    kb_namespace: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        if not self.kb_namespace:
            self.kb_namespace = f"ns_{self.tenant_id.replace('-', '_')}"


class SessionContext(BaseModel):
    """Session context for multi-turn conversations"""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_context: TenantContext
    bot_id: str
    messages: list[MCPMessage] = []
    memory: dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)


# ===== Query & Response =====


class MCPQueryRequest(BaseModel):
    """Request for MCP query processing"""

    query: str
    bot_id: str
    session_id: str | None = None
    stream: bool = False
    context: dict[str, Any] = {}
    tool_options: dict[str, bool] = {}  # Enable/disable specific tools
    custom_prompt: str | None = None  # In-memory system prompt override from Studio
    model: str | None = None  # Per-bot LLM model override (bare id; tenant key/provider still apply)


class ToolCall(BaseModel):
    """Tool call information"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: dict[str, Any] = {}
    status: ToolCallStatus = ToolCallStatus.PENDING
    result: Any | None = None
    error: str | None = None
    duration_ms: int | None = None


class Source(BaseModel):
    """Knowledge source reference"""

    kb_id: str
    kb_name: str
    document_id: str
    document_title: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = {}


class MCPQueryResponse(BaseModel):
    """Response from MCP query processing"""

    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    answer: str
    sources: list[Source] = []
    tool_calls: list[ToolCall] = []
    tokens_used: int = 0
    latency_ms: int = 0
    model: str = ""
    session_id: str = ""


# ===== Provisioning =====


class KBConfig(BaseModel):
    """Knowledge base configuration"""

    name: str
    kb_type: str  # web_crawler, document, confluence, gdrive, etc.
    connector_type: str | None = None
    connector_config: dict[str, Any] = {}
    chunking_config: dict[str, Any] = {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "separator": "\n\n",
    }
    embedding_config: dict[str, Any] = {
        "model": "text-embedding-3-small",
        "dimensions": 1536,
    }


class ProvisionKBRequest(BaseModel):
    """Request to provision a knowledge base"""

    bot_id: str
    kb_config: KBConfig


class ProvisionKBResponse(BaseModel):
    """Response from KB provisioning"""

    kb_id: str
    hello_id: str
    unique_name: str
    status: KBStatus
    message: str


class ProvisionBotRequest(BaseModel):
    """Request to provision a bot"""

    name: str
    description: str = ""
    kb_ids: list[str] = []
    prompt_config: dict[str, Any] = {}
    tool_config: dict[str, bool] = {}


class ProvisionBotResponse(BaseModel):
    """Response from bot provisioning"""

    bot_id: str
    status: BotStatus
    message: str


# ===== Testing =====


class TestQuery(BaseModel):
    """Test query for bot validation"""

    query: str
    expected_topics: list[str] = []
    expected_sources: list[str] = []


class TestResult(BaseModel):
    """Result of a single test"""

    query: str
    response: str
    passed: bool
    issues: list[str] = []
    latency_ms: int = 0


class TestBotRequest(BaseModel):
    """Request to test a bot"""

    bot_id: str
    test_queries: list[TestQuery] = []


class TestBotResponse(BaseModel):
    """Response from bot testing"""

    bot_id: str
    results: list[TestResult]
    passed: bool
    overall_score: float


# ===== Tool Definitions =====


class ToolParameter(BaseModel):
    """Tool parameter definition"""

    name: str
    type: str  # string, number, boolean, array, object
    description: str
    required: bool = False
    default: Any | None = None
    items: dict[str, Any] | None = None  # For array types, specifies item schema


class ToolDefinition(BaseModel):
    """Tool definition for registry"""

    name: str
    description: str
    parameters: list[ToolParameter] = []
    returns: dict[str, str] = {}  # Return type description
    requires_permissions: list[str] = []
    enabled_by_default: bool = True


class ToolExecutionRequest(BaseModel):
    """Request to execute a tool"""

    tool_name: str
    parameters: dict[str, Any] = {}
    context: dict[str, Any] = {}


class ToolExecutionResponse(BaseModel):
    """Response from tool execution"""

    tool_name: str
    result: Any
    success: bool
    error: str | None = None
    duration_ms: int = 0


# ===== Connector Integration =====


class ConnectorSyncRequest(BaseModel):
    """Request to sync a connector"""

    connector_id: str
    kb_id: str
    full_sync: bool = False


class ConnectorSyncResponse(BaseModel):
    """Response from connector sync"""

    job_id: str
    status: str
    documents_found: int = 0
    message: str


class ConnectorStatus(BaseModel):
    """Connector health status"""

    connector_id: str
    connector_type: str
    health: str  # healthy, degraded, offline
    last_sync: datetime | None = None
    documents_indexed: int = 0
    error: str | None = None
