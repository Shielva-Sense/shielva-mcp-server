"""
MCP Protocol - Core Data Models
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import uuid


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
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TenantContext(BaseModel):
    """Tenant isolation context"""
    tenant_id: str
    user_id: str
    user_email: str
    role: str = "Customer_Basic"
    permissions: List[str] = []
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
    messages: List[MCPMessage] = []
    memory: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)


# ===== Query & Response =====

class MCPQueryRequest(BaseModel):
    """Request for MCP query processing"""
    query: str
    bot_id: str
    session_id: Optional[str] = None
    stream: bool = False
    context: Dict[str, Any] = {}
    tool_options: Dict[str, bool] = {}  # Enable/disable specific tools
    custom_prompt: Optional[str] = None  # In-memory system prompt override from Studio


class ToolCall(BaseModel):
    """Tool call information"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: Dict[str, Any] = {}
    status: ToolCallStatus = ToolCallStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None


class Source(BaseModel):
    """Knowledge source reference"""
    kb_id: str
    kb_name: str
    document_id: str
    document_title: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = {}


class MCPQueryResponse(BaseModel):
    """Response from MCP query processing"""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    answer: str
    sources: List[Source] = []
    tool_calls: List[ToolCall] = []
    tokens_used: int = 0
    latency_ms: int = 0
    model: str = ""
    session_id: str = ""


# ===== Provisioning =====

class KBConfig(BaseModel):
    """Knowledge base configuration"""
    name: str
    kb_type: str  # web_crawler, document, confluence, gdrive, etc.
    connector_type: Optional[str] = None
    connector_config: Dict[str, Any] = {}
    chunking_config: Dict[str, Any] = {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "separator": "\n\n"
    }
    embedding_config: Dict[str, Any] = {
        "model": "text-embedding-3-small",
        "dimensions": 1536
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
    kb_ids: List[str] = []
    prompt_config: Dict[str, Any] = {}
    tool_config: Dict[str, bool] = {}


class ProvisionBotResponse(BaseModel):
    """Response from bot provisioning"""
    bot_id: str
    status: BotStatus
    message: str


# ===== Testing =====

class TestQuery(BaseModel):
    """Test query for bot validation"""
    query: str
    expected_topics: List[str] = []
    expected_sources: List[str] = []


class TestResult(BaseModel):
    """Result of a single test"""
    query: str
    response: str
    passed: bool
    issues: List[str] = []
    latency_ms: int = 0


class TestBotRequest(BaseModel):
    """Request to test a bot"""
    bot_id: str
    test_queries: List[TestQuery] = []


class TestBotResponse(BaseModel):
    """Response from bot testing"""
    bot_id: str
    results: List[TestResult]
    passed: bool
    overall_score: float


# ===== Tool Definitions =====

class ToolParameter(BaseModel):
    """Tool parameter definition"""
    name: str
    type: str  # string, number, boolean, array, object
    description: str
    required: bool = False
    default: Optional[Any] = None
    items: Optional[Dict[str, Any]] = None  # For array types, specifies item schema



class ToolDefinition(BaseModel):
    """Tool definition for registry"""
    name: str
    description: str
    parameters: List[ToolParameter] = []
    returns: Dict[str, str] = {}  # Return type description
    requires_permissions: List[str] = []
    enabled_by_default: bool = True


class ToolExecutionRequest(BaseModel):
    """Request to execute a tool"""
    tool_name: str
    parameters: Dict[str, Any] = {}
    context: Dict[str, Any] = {}


class ToolExecutionResponse(BaseModel):
    """Response from tool execution"""
    tool_name: str
    result: Any
    success: bool
    error: Optional[str] = None
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
    last_sync: Optional[datetime] = None
    documents_indexed: int = 0
    error: Optional[str] = None
