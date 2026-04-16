from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger(__name__)

from config.settings import get_settings

class BotRegistry:
    """
    Registry for bot configurations.
    Fetches actual bot data from MongoDB.
    """
    def __init__(self, mongodb_client=None):
        self.mongodb_client = mongodb_client
        self.settings = get_settings()
        
    async def get_bot(self, bot_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        Get bot configuration by ID and tenant from CustomerProfile.customerService.
        """
        if not self.mongodb_client:
            logger.warning("MongoDB client not initialized in BotRegistry")
            return self._get_mock_bot(bot_id)

        try:
            db = self.mongodb_client[self.settings.mongodb_db_name]
            collection = db.customerService # Collection name from ShielvaAPI logic
            
            # Find customer by tenant_id
            customer = await collection.find_one({
                "tenant_id": tenant_id
            })
            
            if not customer:
                logger.warning("Customer not found for tenant", tenant_id=tenant_id)
                return self._get_mock_bot(bot_id)
                
            # Find specific bot in bots array
            bots = customer.get("bots", [])
            bot = next((b for b in bots if b.get("id") == bot_id), None)
            
            if not bot:
                logger.warning("Bot not found in customer profile", bot_id=bot_id)
                return self._get_mock_bot(bot_id)
            
            # Ensure kb_ids is present and formatted correctly
            # ShielvaAPI might store them as 'kbs' or 'kb_ids'
            # In ShielvaAPI/api/bots.py it pushes to 'bots.$.kbs'
            
            # Extract KB IDs correctly
            kb_ids = []
            if "kbs" in bot and isinstance(bot["kbs"], list):
                for kb in bot["kbs"]:
                    if isinstance(kb, dict):
                        kb_ids.append(kb.get("id"))
                    else:
                        kb_ids.append(kb)
            elif "kb_ids" in bot:
                kb_ids = bot.get("kb_ids", [])
                
            bot["kb_ids"] = kb_ids
            
            # Log for debugging
            logger.info("Fetched bot config", bot_id=bot_id, name=bot.get("name"), kb_count=len(kb_ids))
            
            return bot
            
        except Exception as e:
            logger.error("Error fetching bot from MongoDB", error=str(e))
            return self._get_mock_bot(bot_id)

    def _get_mock_bot(self, bot_id: str) -> Dict[str, Any]:
        """Fallback mock bot"""
        return {
            "id": bot_id,
            "name": f"Bot {bot_id} (Fallback)",
            "description": "Auto-generated bot configuration (DB Error)",
            "prompt_config": {
                "system_prompt": "You are a helpful AI assistant. Answer questions clearly and accurately based on the provided context.",
                "tool_instructions": "Use available tools if the user question requires specialized actions."
            },
            "kb_ids": [],
            "model_config": {
                "model": "gemini-1.5-pro",
                "temperature": 0.1
            }
        }
