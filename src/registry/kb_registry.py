from typing import Dict, Any, List, Optional
import structlog
import uuid
from src.protocol.models import TenantContext

logger = structlog.get_logger(__name__)

class KBInfo:
    def __init__(self, kb_id, hello_id, unique_name, status):
        self.kb_id = kb_id
        self.hello_id = hello_id
        self.unique_name = unique_name
        self.status = status

class KBRegistry:
    """
    Registry for knowledge base configurations.
    """
    def __init__(self, mongodb_client=None):
        self.mongodb_client = mongodb_client
    
    async def register(self, tenant_context: TenantContext, bot_id: str, kb_config: Any) -> KBInfo:
        """
        Register a new KB.
        """
        kb_id = str(uuid.uuid4())
        hello_id = f"hello_{kb_id[:8]}"
        unique_name = f"{kb_config.name.lower().replace(' ', '_')}_{kb_id[:4]}"
        
        logger.info(
            "Registering KB",
            tenant_id=tenant_context.tenant_id,
            bot_id=bot_id,
            kb_id=kb_id
        )
        
        return KBInfo(
            kb_id=kb_id,
            hello_id=hello_id,
            unique_name=unique_name,
            status="ready"
        )
    
    async def get_kbs_for_bot(self, bot_id: str, tenant_id: str) -> List[Dict[str, Any]]:
        """
        Get all KBs for a bot.
        """
        return []
