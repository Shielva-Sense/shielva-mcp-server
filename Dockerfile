# ─────────────────────────────────────────────────────────────────────
# shielva-mcp-server — production container image.
#
# Multi-stage build:
#   stage 1  builder  Python 3.11-slim + build toolchain. Installs
#                     deps from requirements.txt (includes the
#                     `shielva-common` GitHub release) into an
#                     isolated /opt/venv.
#   stage 2  runtime  Python 3.11-slim, non-root UID 1000, copies
#                     /opt/venv + src/ + config/ + healthcheck.py.
#
# Build context:
#
#   The build context is the component directory itself. Canonical:
#
#     cd shielva-mcp/shielva-mcp/mcp-server
#     docker build -t shielva-mcp-server:dev .
#
# Hardening (SOC2 CC6.8):
#   * non-root UID/GID 1000 (`appuser`)
#   * runtime image declares no SUID/SGID writable paths
#   * orchestrator-side: read_only=true, cap_drop=ALL,
#     seccomp=runtime/default, allowPrivilegeEscalation=false
#   * PYTHONDONTWRITEBYTECODE=1 — runtime writes no .pyc to /srv
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# Stage 1 — builder
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Build-only toolchain. Track Debian stable's stream rather than pinning
# each apt version: bookworm-security pushes silently and pinning makes
# the image fail to rebuild after a CVE patch — opposite of CC7.1.
# `git` is required so pip can resolve the `shielva-common @ git+...`
# entry in requirements.txt.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
        g++ \
        pkg-config \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
RUN python -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip==24.3.1 setuptools==75.6.0 wheel==0.45.1

WORKDIR /build

# Copy requirements first so the layer cache survives source-only
# changes — the expensive resolve only re-runs when deps change.
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Sanity gate — fail the build if core imports cannot resolve.
RUN python -c "import fastapi, uvicorn, structlog, httpx; print('builder ok')"

# ─────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
# ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime needs ca-certificates (for httpx TLS verification against
# internal CA). curl is intentionally OMITTED — healthcheck is a
# Python stdlib script so we don't carry an extra binary.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. Matches Helm `runAsUser: 1000`.
RUN groupadd --system --gid 1000 appuser \
    && useradd --system --uid 1000 --gid 1000 \
        --home-dir /home/appuser --create-home --shell /usr/sbin/nologin \
        appuser

# Resolved venv. Owned by root: runtime user only needs read.
COPY --from=builder /opt/venv /opt/venv

# Application code. Owned by root so a compromised runtime user cannot
# mutate code at runtime — readOnlyRootFilesystem is the belt; this is
# the braces.
WORKDIR /srv
COPY --chown=root:root src       /srv/src
COPY --chown=root:root config    /srv/config
COPY --chown=root:root schedule_config.json /srv/schedule_config.json

# Healthcheck probe — stdlib-only so a broken httpx won't break health.
COPY --chown=root:root --chmod=0555 scripts/healthcheck.py /usr/local/bin/healthcheck.py

# Writable runtime path. Audit JSONL backups + transient state go under
# /var/lib/shielva-mcp-server (FHS-compliant for service state).
RUN mkdir -p /var/lib/shielva-mcp-server \
    && chown -R appuser:appuser /var/lib/shielva-mcp-server \
    && chmod 0750 /var/lib/shielva-mcp-server

# Runtime env defaults. SOP_ENABLED is off by default so a smoke
# `docker run` does not try to dial a non-existent collector.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv \
    PYTHONPATH=/srv \
    SOP_ENABLED=false \
    HOST=0.0.0.0 \
    MCP_PORT=8004 \
    PORT=8004

EXPOSE 8004

USER appuser:appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python3", "/usr/local/bin/healthcheck.py"]

# 4 workers — calibrated against the platform reference image. Capacity
# tuning uses N+1 horizontal pods, not more workers per pod.
CMD ["python3", "-m", "uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8004", \
     "--workers", "4", \
     "--backlog", "2048", \
     "--no-access-log", \
     "--log-level", "warning"]
