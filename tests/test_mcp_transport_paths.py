"""The protocol endpoint answers at the router root as well as at /mcp.

🚨 The gateway routes /api/mcp/* to this service and strips /api/mcp, so a
client pasting the endpoint into its config had to write /api/mcp/mcp — the
word twice, once from the gateway's route and once from ours. Serving the same
handler at the root makes /api/mcp/ the clean URL without breaking anything
already pointed at /mcp.
"""

from __future__ import annotations

import inspect

from src.interface.mcp_jsonrpc import transport


def test_the_protocol_is_served_at_both_paths():
    src = inspect.getsource(transport.build_router)
    assert '_PROTOCOL_PATHS = ("/mcp", "/")' in src


def test_every_verb_registers_both_paths():
    """A verb wired to only one of them is an endpoint that works on one URL
    and 404s on the other — the confusing half-fix."""
    src = inspect.getsource(transport.build_router)
    for verb in ("post", "delete", "get"):
        assert src.count(f"@router.{verb}(_PROTOCOL_PATHS[0]") == 1, verb
        assert src.count(f"@router.{verb}(_PROTOCOL_PATHS[1]") == 1, verb


def test_only_one_of_the_pair_is_documented():
    """Both in the schema would render every MCP operation twice in the docs."""
    src = inspect.getsource(transport.build_router)
    assert src.count("include_in_schema=False") == 3
