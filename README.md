# Shielva MCP Server

## Overview

The MCP (Model Context Protocol) Server is the **AI Operating System** of Shielva ARC. It orchestrates knowledge retrieval, tool execution, and connector coordination for multi-tenant AI bots.

## Quick Start

```bash
cd mcp-server
pip install -r requirements.txt
cp .env.example .env
python -m src.main
```

## Architecture

```
MCP Server
├── Protocol Layer     → Message handling, session management
├── Context Layer      → Context assembly, prompt engineering
├── Registry Layer     → Tool & KB registration
├── Routing Layer      → LLM routing, tool dispatch
├── Lifecycle Layer    → Provisioning, testing
├── Security Layer     → Policy enforcement, sandbox
├── Transport Layer    → REST/gRPC handlers
└── Ingest API         → RAG ingestion management (security-fix-collector)
```

## Key Features

- **LangChain Agent Orchestration** - ReAct agents with tool calling
- **Multi-Tenant Isolation** - Namespace-based vector DB isolation
- **Connector Integration** - Unified interface for 25+ data sources
- **RAG Pipeline** - Hybrid search with reranking
- **LLM Abstraction** - LiteLLM for multi-provider support
- **Policy Engine** - OPA-based RBAC and quotas
- **Security Fix RAG** - `security_fix_query` tool for CVE fix-diff retrieval
- **Ingestion API** - HTTP endpoints to trigger/monitor the security-fix-collector

## Technology Stack

- **Framework**: FastAPI
- **Agent**: LangChain + LangGraph
- **RAG**: LlamaIndex
- **Vector DB**: Supabase pgvector (`security_fix_entries` table)
- **LLM Router**: LiteLLM
- **Cache**: Redis
- **Queue**: Celery

---

## RAG Ingestion API

The MCP server exposes a management API for the `security-fix-collector` pipeline. These endpoints are consumed by `shielva-security-api` settings endpoints — the security UI never calls MCP directly.

**Base prefix:** `/mcp/v1/ingest`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/mcp/v1/ingest/stats` | Vector-store stats: total entries, collector availability. **60 s TTL cached.** |
| `GET` | `/mcp/v1/ingest` | List last 10 ingestion jobs (most-recent first) |
| `POST` | `/mcp/v1/ingest/start` | Start a bulk ingestion job. Returns `202 Accepted` with `job_id`. |
| `GET` | `/mcp/v1/ingest/{job_id}` | Poll a specific job — progress counters, current ecosystem, rolling log tail |
| `POST` | `/mcp/v1/ingest/{job_id}/cancel` | Terminate the collector subprocess and mark job `cancelled` |

### Start request body

```json
{
  "ecosystems": ["npm", "PyPI", "Go", "Maven", "RubyGems", "NuGet"],
  "limit": 10000,
  "github_token": "ghp_..."
}
```

- `ecosystems` — subset of the six supported; defaults to all six.
- `limit` — max advisories per run (100 – 100,000). Keep at 10k for initial seed; increase for deeper coverage.
- `github_token` — optional. Without it: 60 req/hr (unauthenticated GitHub API). With it: 5,000 req/hr. Passed from the tenant's stored PAT via the security API proxy — never stored in MCP.

### Job model

```json
{
  "job_id": "uuid",
  "status": "pending | running | completed | failed | cancelled",
  "ecosystems": ["PyPI"],
  "limit": 10000,
  "started_at": "2026-04-15T12:00:00Z",
  "completed_at": null,
  "current_ecosystem": "PyPI",
  "advisories_processed": 342,
  "entries_stored": 287,
  "error_message": null,
  "logs": ["last 200 stdout lines from collector"]
}
```

### How it works

The MCP server spawns `security-fix-collector/collector.py --mode bulk` as an async subprocess. Stdout is streamed line-by-line; the server parses:

- `── Processing ecosystem: X ──` → updates `current_ecosystem`
- `total: N` → updates `entries_stored`
- `advisories processed: N` → updates `advisories_processed`
- Exit code 0 → `completed`; non-zero → `failed`

Job state is held in-process (Python dict). On cancel, the subprocess receives `SIGTERM`. There is no persistence — jobs are lost on MCP server restart.

### Proxy chain

```
Security UI
  └── PATCH /security/api/v1/settings/rag/ingest
        └── Security API (settings.py)
              └── POST https://localhost:8004/mcp/v1/ingest/start
                    └── MCP Server (ingest_routes.py)
                          └── subprocess: python3 collector.py --mode bulk
```

The security API decrypts the tenant GitHub PAT (Fernet) before forwarding it. The token is never stored in MCP or logged.

---

## Security Fix Query Tool

Registered as `security_fix_query` in the tool registry. Called by `shielva-security-api` during the AI enrichment phase after each scan.

**Vector table:** `security_fix_entries` (Supabase pgvector, 768-dim Gemini embeddings)

**Query-time flow:**
1. Build query text from `cve_id`, `cwe_id`, `ecosystem`, `package`, `severity`, `code_snippet`
2. Embed with Gemini `embedding-001` (same model used during ingestion)
3. Cosine similarity search via `<=>` pgvector operator
4. Return top-K results (default 5) above 0.45 similarity threshold

**Coverage notes:**
- SCA findings with a CVE ID → high-precision match
- SAST/Bandit findings (no CVE) → CWE + code-snippet similarity, lower precision
- Empty store → enrichment step skips gracefully, no fix suggestions generated

---

## API Reference

See `api_reference.md` for full endpoint documentation.
