"""
Shielva Assist — Live Tenant Context Builder

Fetches real tenant data from CMS Core to give the LLM
accurate, up-to-date context about the user's configuration.
"""
import os
from typing import Optional, Dict, Any
import httpx
import structlog

from shielva_common.tls import internal_ca_verify

logger = structlog.get_logger(__name__)

# CMS Core base URL (internal, through gateway or direct)
CMS_CORE_URL = os.getenv("CMS_CORE_URL", "https://localhost:8020/api/v1/cms")


async def fetch_tenant_context(tenant_id: str, context_page: Optional[str] = None) -> str:
    """
    Fetch live tenant data and format as context for the LLM.

    Returns a concise summary string injected into the system prompt.
    """
    if not tenant_id:
        return "No tenant selected."

    sections = []
    async with httpx.AsyncClient(verify=internal_ca_verify(), timeout=5.0) as client:
        # Fetch intents
        try:
            r = await client.get(f"{CMS_CORE_URL}/intents", params={"tenant_id": tenant_id})
            if r.status_code == 200:
                intents = r.json()
                enabled = [i for i in intents if i.get("enabled")]
                disabled = [i for i in intents if not i.get("enabled")]
                if intents:
                    intent_list = ", ".join(
                        f"`{i.get('intent_id', '?')}` ({i.get('label', '?')}, {len(i.get('examples', []))} examples)"
                        for i in intents[:20]
                    )
                    sections.append(
                        f"**Intents**: {len(intents)} total ({len(enabled)} enabled, {len(disabled)} disabled)\n{intent_list}"
                    )
                else:
                    sections.append("**Intents**: None configured yet.")
        except Exception as e:
            logger.debug("Failed to fetch intents", error=str(e))

        # Fetch automations
        try:
            r = await client.get(f"{CMS_CORE_URL}/automations", params={"tenant_id": tenant_id})
            if r.status_code == 200:
                automations = r.json()
                sections.append(f"**Automations**: {len(automations)} configured")
        except Exception:
            pass

        # Fetch signal rules
        try:
            r = await client.get(f"{CMS_CORE_URL}/signals/rules", params={"tenant_id": tenant_id})
            if r.status_code == 200:
                rules = r.json()
                sections.append(f"**Signal Rules**: {len(rules)} configured")
        except Exception:
            pass

        # Fetch decision logic
        try:
            r = await client.get(f"{CMS_CORE_URL}/logic")
            if r.status_code == 200:
                logic = r.json()
                sections.append(f"**Decision Rules**: {len(logic)} configured")
        except Exception:
            pass

    if not sections:
        return f"Tenant `{tenant_id}` — could not fetch live data."

    return f"## Live Tenant Data ({tenant_id})\n" + "\n".join(sections)
