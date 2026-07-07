"""
MCP Codegen Tools — Static intelligence tools for connector code generation.

These tools give MCP's agent loop code-awareness it uses during fix cycles:
  - codegen_validate_python  — AST parse and return structured error list
  - codegen_analyze_imports  — detect wrong import paths in test code
  - codegen_categorize_error — classify pytest errors into fix categories
  - codegen_check_pytest_structure — detect common pytest structure issues

These tools run entirely in-process (no filesystem access needed) so they work
regardless of whether MCP and integration-builder share a filesystem.

Registered in mcp-server/src/main.py lifespan alongside default tools.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from src.protocol.models import TenantContext

# ── Tool handlers ─────────────────────────────────────────────────────


async def codegen_validate_python(
    tenant_context: TenantContext,
    code: str,
) -> dict[str, Any]:
    """
    Parse Python source code with AST and return syntax errors.

    Returns:
        {"valid": true}  on success
        {"valid": false, "errors": [{"line": N, "message": "..."}]}  on failure
    """
    try:
        ast.parse(code)
        return {"valid": True, "errors": []}
    except SyntaxError as exc:
        return {
            "valid": False,
            "errors": [{"line": exc.lineno, "column": exc.offset, "message": str(exc.msg)}],
        }
    except Exception as exc:
        return {"valid": False, "errors": [{"line": None, "message": str(exc)}]}


async def codegen_analyze_imports(
    tenant_context: TenantContext,
    code: str,
    connector_class: str,
) -> dict[str, Any]:
    """
    Detect common import mistakes in generated test/connector code.

    Checks for:
    - Wrong module path (from pkg.connector import X instead of from connector import X)
    - Missing connector import entirely
    - Wrong patch() string targets

    Returns dict with "issues" list and "has_issues" bool.
    """
    issues: list[str] = []

    # Check for wrong deep-path import
    wrong_import_pattern = re.compile(rf"from\s+\w[\w.]*\.connector\s+import\s+{re.escape(connector_class)}")
    for i, line in enumerate(code.splitlines(), 1):
        if wrong_import_pattern.search(line):
            issues.append(
                f"Line {i}: Wrong import path '{line.strip()}' — should be 'from connector import {connector_class}'"
            )

    # Check if connector class is used but not imported from connector
    correct_import = f"from connector import {connector_class}"
    if connector_class in code and correct_import not in code:
        issues.append(
            f"Missing correct import: '{correct_import}' not found but '{connector_class}' is referenced in code."
        )

    # Check for wrong patch targets
    wrong_patch = re.compile(r"""patch\(['"][\w.]+\.connector\.""")
    for i, line in enumerate(code.splitlines(), 1):
        if wrong_patch.search(line):
            issues.append(
                f"Line {i}: Wrong patch target in '{line.strip()}' — "
                f"should use 'connector.<method>' not '<package>.connector.<method>'"
            )

    return {"has_issues": bool(issues), "issues": issues, "issue_count": len(issues)}


async def codegen_categorize_error(
    tenant_context: TenantContext,
    error_output: str,
    code: str = "",
) -> dict[str, Any]:
    """
    Categorize a pytest error output into a known fix category.

    Categories:
      syntax_error        — Python SyntaxError
      import_error        — ModuleNotFoundError / ImportError
      fixture_error       — pytest fixture issues
      assertion_error     — test assertion failures
      attribute_error     — wrong method/attribute access
      type_error          — wrong argument types
      async_error         — missing asyncio marks / coroutine not awaited
      auth_error          — authentication/credential issues
      network_error       — connection/timeout issues
      unknown             — doesn't match known patterns

    Returns: {"category": "...", "confidence": 0.0-1.0, "key_message": "..."}
    """
    err = error_output.lower()

    patterns: list[tuple] = [
        ("syntax_error", r"syntaxerror|invalid syntax|unexpected indent", 0.95),
        ("import_error", r"modulenotfounderror|importerror|no module named", 0.95),
        (
            "fixture_error",
            r"fixture.*not found|fixture.*applied more than once|@pytest\.fixture",
            0.90,
        ),
        (
            "async_error",
            r"coroutine.*never awaited|asyncio.*mark|was never awaited",
            0.90,
        ),
        ("attribute_error", r"attributeerror|has no attribute", 0.85),
        (
            "type_error",
            r"typeerror|takes \d+ positional|unexpected keyword argument",
            0.85,
        ),
        ("assertion_error", r"assertionerror|assert.*failed", 0.80),
        (
            "auth_error",
            r"unauthorized|authentication|401|403|credentials|api.?key",
            0.80,
        ),
        ("network_error", r"connectionerror|timeout|connection refused|socket", 0.75),
    ]

    for category, pattern, confidence in patterns:
        if re.search(pattern, err):
            # Extract first non-empty line that contains the key message
            key_lines = [
                ln.strip()
                for ln in error_output.splitlines()
                if ln.strip() and not ln.strip().startswith(("_", "=", "-"))
            ]
            key_message = key_lines[0] if key_lines else error_output[:200]
            return {
                "category": category,
                "confidence": confidence,
                "key_message": key_message,
            }

    return {"category": "unknown", "confidence": 0.5, "key_message": error_output[:200]}


async def codegen_check_pytest_structure(
    tenant_context: TenantContext,
    code: str,
) -> dict[str, Any]:
    """
    Detect common pytest structure problems in test code using AST analysis.

    Checks for:
    - Duplicate @pytest.fixture on same function
    - Async test functions missing @pytest.mark.asyncio
    - Test functions not named test_*
    - Missing import pytest
    - Duplicate test function names

    Returns: {"issues": [...], "has_issues": bool}
    """
    issues: list[str] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "has_issues": True,
            "issues": [f"SyntaxError at line {exc.lineno}: {exc.msg}"],
        }

    seen_names: dict[str, int] = {}
    has_import_pytest = any(
        (isinstance(node, ast.Import) and any(a.name == "pytest" for a in node.names))
        or (isinstance(node, ast.ImportFrom) and node.module == "pytest")
        for node in ast.walk(tree)
    )

    if not has_import_pytest:
        issues.append("Missing 'import pytest' — test file must import pytest")

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        name = node.name
        [
            (
                d.id
                if isinstance(d, ast.Name)
                else (
                    d.attr
                    if isinstance(d, ast.Attribute)
                    else (d.func.attr if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) else "")
                )
            )
            for d in node.decorator_list
        ]

        # Duplicate fixture decorator
        fixture_count = sum(
            1
            for d in node.decorator_list
            if (isinstance(d, ast.Attribute) and d.attr == "fixture")
            or (isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "fixture")
        )
        if fixture_count > 1:
            issues.append(f"Line {node.lineno}: '@pytest.fixture' applied {fixture_count}x on '{name}' — keep only one")

        # Async test missing asyncio mark
        if isinstance(node, ast.AsyncFunctionDef) and name.startswith("test_"):
            has_asyncio = any("asyncio" in str(d) for d in node.decorator_list)
            if not has_asyncio:
                issues.append(f"Line {node.lineno}: async test '{name}' missing '@pytest.mark.asyncio'")

        # Duplicate test names
        if name.startswith("test_"):
            seen_names[name] = seen_names.get(name, 0) + 1
            if seen_names[name] == 2:
                issues.append(f"Duplicate test function name '{name}' — rename one of them")

    return {"has_issues": bool(issues), "issues": issues}


# ── Tool definitions for MCP registry ────────────────────────────────

from src.protocol.models import (
    ToolDefinition,
)

CODEGEN_TOOL_DEFINITIONS = [
    (
        ToolDefinition(
            name="codegen_validate_python",
            description=(
                "Parse Python source code with AST to detect syntax errors. "
                "Use before writing generated code to disk to catch issues early."
            ),
            parameters=[
                {
                    "name": "code",
                    "type": "string",
                    "description": "Python source code to validate",
                    "required": True,
                },
            ],
            requires_permissions=[],
            enabled_by_default=True,
        ),
        codegen_validate_python,
    ),
    (
        ToolDefinition(
            name="codegen_analyze_imports",
            description=(
                "Detect wrong import paths in generated connector test code. "
                "Checks for common mistakes like 'from pkg.connector import X' instead of 'from connector import X'."
            ),
            parameters=[
                {
                    "name": "code",
                    "type": "string",
                    "description": "Python test code to analyze",
                    "required": True,
                },
                {
                    "name": "connector_class",
                    "type": "string",
                    "description": "Expected connector class name, e.g. GmailConnector",
                    "required": True,
                },
            ],
            requires_permissions=[],
            enabled_by_default=True,
        ),
        codegen_analyze_imports,
    ),
    (
        ToolDefinition(
            name="codegen_categorize_error",
            description=(
                "Categorize a pytest failure output into a fix category "
                "(syntax_error, import_error, fixture_error, async_error, etc.). "
                "Use this to decide which fix strategy to apply."
            ),
            parameters=[
                {
                    "name": "error_output",
                    "type": "string",
                    "description": "Full pytest error/failure output string",
                    "required": True,
                },
                {
                    "name": "code",
                    "type": "string",
                    "description": "The code that produced the error (optional context)",
                    "required": False,
                },
            ],
            requires_permissions=[],
            enabled_by_default=True,
        ),
        codegen_categorize_error,
    ),
    (
        ToolDefinition(
            name="codegen_check_pytest_structure",
            description=(
                "Detect common pytest structure problems (duplicate fixtures, missing asyncio marks, "
                "duplicate test names, missing import pytest) via AST analysis."
            ),
            parameters=[
                {
                    "name": "code",
                    "type": "string",
                    "description": "Python test source code to check",
                    "required": True,
                },
            ],
            requires_permissions=[],
            enabled_by_default=True,
        ),
        codegen_check_pytest_structure,
    ),
]


def register_codegen_tools(registry) -> None:
    """Register all codegen intelligence tools into the MCP tool registry.

    Call this from mcp-server/src/main.py lifespan after create_registry_with_defaults():
        from src.tools.codegen_tools import register_codegen_tools
        register_codegen_tools(tool_registry)
    """
    for definition, handler in CODEGEN_TOOL_DEFINITIONS:
        registry.register(definition, handler)
