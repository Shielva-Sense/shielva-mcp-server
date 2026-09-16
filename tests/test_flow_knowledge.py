"""What the guide promises must match what the runtime actually does."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.protocol.models import TenantContext
from src.tools import flow_knowledge as fk

_RUNTIME = Path("/Volumes/V3-SSD/Shielva Project Dirs/shielva-api/app/services/flow_runtime.py")


def _t():
    return TenantContext(tenant_id="t1", user_id="u1", user_email="u@example.com")


def test_the_capability_table_matches_the_runtime():
    """🚨 A copy that drifts is worse than no copy, because a model trusts it.
    core-api's _CAPABILITY_OF is the authority; if somebody adds a fifth
    capability node there, this fails rather than quietly telling models the
    wrong thing."""
    if not _RUNTIME.exists():
        return  # core-api not checked out beside this repo
    src = _RUNTIME.read_text()
    block = src[src.index("_CAPABILITY_OF = {") : src.index("_CAPABILITY_KINDS")]
    for kind, capability in fk.CAPABILITY_OF_KIND.items():
        assert f'"{kind}": "{capability}"' in block, f"{kind} drifted from the runtime"
    # And nothing in the runtime is missing from ours.
    import re

    runtime_kinds = set(re.findall(r'"([a-z_]+)":\s*"[a-z_.]+"', block))
    assert runtime_kinds == set(fk.CAPABILITY_OF_KIND), (
        f"runtime has {runtime_kinds}, guide has {set(fk.CAPABILITY_OF_KIND)}"
    )


def test_every_documented_kind_is_one_the_runtime_recognises():
    """A kind the runtime does not match saves cleanly and never executes, so
    documenting an invented one is worse than omitting it."""
    if not _RUNTIME.exists():
        return
    src = _RUNTIME.read_text()
    for entry in fk.NODE_KINDS:
        assert f'"{entry["kind"]}"' in src, f"{entry['kind']} is not known to flow_runtime"


def test_the_guide_leads_with_configure_before_nodes():
    """🚨 The owner's rule. A capability node added before its capability is
    enabled saves fine and does nothing at runtime, with no error anywhere —
    the runtime's own comment records that exact bug reaching production."""
    guide = asyncio.run(fk.shielva_flow_guide(_t()))
    assert guide["build_order"][0]["do"].lower().startswith("configure the bot first")
    assert "capability" in guide["build_order"][0]["why"].lower()

    for row in guide["capability_nodes"]:
        assert "does nothing at runtime" in row["if_skipped"]
        assert "shielva_set_bot_capability" in row["enable_with"]


def test_nodes_that_point_at_something_say_what_must_exist_first():
    by_kind = {n["kind"]: n for n in fk.NODE_KINDS}
    assert "action schema" in by_kind["action"]["requires"]
    assert "intents" in by_kind["classify"]["requires"]
    assert "painter" in by_kind["painter"]["requires"]
    assert "knowledge base" in by_kind["answer"]["requires"]


def test_the_guide_tool_is_described_as_read_this_first():
    """A model only reads tool descriptions; if this one does not say to read it
    before building, it will not be read before building."""
    definition, _ = fk.FLOW_KNOWLEDGE_TOOL_DEFINITIONS[0]
    assert definition.description.startswith("READ THIS FIRST")
    assert "inert until" in definition.description
