"""Tool adapters — implementations of the domain tools ports.

Slice 2 ships :class:`LegacyToolRegistryAdapter`, which wraps the
existing ``src.registry.tool_registry.ToolRegistry`` so both the
new use-case path AND the legacy callers (codegen fix-agent,
message_handler) keep working from the same underlying tool set.

Once slice 4 lifts the remaining legacy callers, the legacy
registry will be deleted and replaced by a clean adapter that
registers tools through the new port directly.
"""

from .legacy_registry_adapter import LegacyToolRegistryAdapter

__all__ = ["LegacyToolRegistryAdapter"]
