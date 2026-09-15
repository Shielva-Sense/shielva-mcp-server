"""The server advertises OAuth so a client offers "Sign in", not a key box.

🚨 An MCP client cannot guess that an authorization server exists. RFC 9728 is
the only channel: a metadata document at the resource, and a 401 whose
``WWW-Authenticate`` names that document. Drop either half and Claude Desktop
falls back to asking the user to paste an API key — which is exactly the
symptom this guards against.

🚨 The document must land on the PUBLIC url. The gateway routes /api/mcp/* here
and strips /api/mcp, so the router has to serve the well-known path at the
service ROOT — a prefix of its own would push the public url to
/api/mcp/<prefix>/.well-known/... where no client looks.
"""

from __future__ import annotations

import inspect
import re

import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.testclient import TestClient

from config.settings import get_settings
from src.core.error_handlers import install_exception_handlers
from src.interface.mcp_jsonrpc import protected_resource
from src.interface.mcp_jsonrpc import transport as transport_mod


@pytest.fixture
def fresh_settings(monkeypatch):
    """Re-read settings under explicit env. ``get_settings`` is lru_cached, so
    a test that only sets env would read whatever the first caller cached."""

    def _apply(**env: str):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        get_settings.cache_clear()
        return get_settings()

    yield _apply
    get_settings.cache_clear()


# ── the document ─────────────────────────────────────────────────────


def test_the_well_known_path_is_the_rfc_constant():
    assert protected_resource.WELL_KNOWN_PATH == "/.well-known/oauth-protected-resource"


def test_the_document_has_every_field_a_client_reads(fresh_settings):
    fresh_settings(
        MCP_OAUTH_RESOURCE_URL="https://gw.example.test/api/mcp",
        MCP_OAUTH_AUTHORIZATION_SERVER="https://gw.example.test/identity",
        MCP_OAUTH_SCOPES="openid,profile,email",
    )
    doc = protected_resource.metadata_document()

    assert doc["resource"] == "https://gw.example.test/api/mcp"
    assert doc["authorization_servers"] == ["https://gw.example.test/identity"]
    assert doc["bearer_methods_supported"] == ["header"]
    assert doc["scopes_supported"] == ["openid", "profile", "email"]


def test_the_urls_come_from_config_not_a_literal(fresh_settings):
    """Point the env somewhere else and the document must follow. A literal
    would keep reporting the old host and the client would sign in to it."""
    fresh_settings(
        MCP_OAUTH_RESOURCE_URL="https://other.example.test/api/mcp",
        MCP_OAUTH_AUTHORIZATION_SERVER="https://idp.example.test/identity",
    )
    doc = protected_resource.metadata_document()

    assert doc["resource"] == "https://other.example.test/api/mcp"
    assert doc["authorization_servers"] == ["https://idp.example.test/identity"]


def test_no_environment_specific_host_is_hardcoded():
    """A deployed hostname in source is the bug this rule exists for — it
    would survive into every other environment."""
    for mod in (protected_resource, transport_mod):
        src = inspect.getsource(mod)
        assert "api.shielva.ai" not in src, mod.__name__
        # Any absolute https:// url outside a docstring/comment would be a
        # baked-in environment. The module may name none.
        code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
        code = re.sub(r'""".*?"""', "", code, flags=re.S)
        assert "https://" not in code, mod.__name__


def test_a_trailing_slash_never_doubles_up(fresh_settings):
    fresh_settings(
        MCP_OAUTH_RESOURCE_URL="https://gw.example.test/api/mcp/",
        MCP_OAUTH_AUTHORIZATION_SERVER="https://gw.example.test/identity/",
    )
    doc = protected_resource.metadata_document()

    assert doc["resource"] == "https://gw.example.test/api/mcp"
    assert doc["authorization_servers"] == ["https://gw.example.test/identity"]
    assert protected_resource.resource_metadata_url() == (
        "https://gw.example.test/api/mcp/.well-known/oauth-protected-resource"
    )


def test_scopes_are_comma_separated_not_json(fresh_settings):
    """The env value is written by hand into a k8s env block; demanding JSON
    there is how the list silently ends up empty."""
    fresh_settings(MCP_OAUTH_SCOPES=" openid , email ,, ")
    assert protected_resource.scopes_supported() == ["openid", "email"]


# ── the route ────────────────────────────────────────────────────────


def test_the_route_is_served_at_the_service_root(fresh_settings):
    """No router prefix: the gateway's /api/mcp strip is what supplies the
    public prefix, so anything extra here moves the document off the url
    clients probe."""
    fresh_settings(
        MCP_OAUTH_RESOURCE_URL="https://gw.example.test/api/mcp",
        MCP_OAUTH_AUTHORIZATION_SERVER="https://gw.example.test/identity",
    )
    router = protected_resource.build_router()
    paths = [r.path for r in router.routes]
    assert paths == ["/.well-known/oauth-protected-resource"]


def test_the_document_is_reachable_without_credentials(fresh_settings):
    """A metadata document behind auth teaches a client nothing — it has no
    credential yet, which is the entire reason it is asking."""
    fresh_settings(
        MCP_OAUTH_RESOURCE_URL="https://gw.example.test/api/mcp",
        MCP_OAUTH_AUTHORIZATION_SERVER="https://gw.example.test/identity",
    )
    app = FastAPI()
    app.include_router(protected_resource.build_router())

    resp = TestClient(app).get("/.well-known/oauth-protected-resource")

    assert resp.status_code == 200
    body = resp.json()
    assert body["resource"] == "https://gw.example.test/api/mcp"
    assert body["authorization_servers"] == ["https://gw.example.test/identity"]


# ── the 401 challenge ────────────────────────────────────────────────


def test_the_challenge_names_the_metadata_document(fresh_settings):
    fresh_settings(MCP_OAUTH_RESOURCE_URL="https://gw.example.test/api/mcp")
    value = protected_resource.www_authenticate_value()

    assert value == ('Bearer resource_metadata="https://gw.example.test/api/mcp/.well-known/oauth-protected-resource"')


def test_the_challenge_points_at_the_same_resource_the_document_reports(fresh_settings):
    """Two independent readers of the url would drift; a client following a
    challenge to a document that disagrees learns nothing it can use."""
    fresh_settings(MCP_OAUTH_RESOURCE_URL="https://gw.example.test/api/mcp")
    doc_resource = protected_resource.metadata_document()["resource"]

    assert doc_resource + protected_resource.WELL_KNOWN_PATH in protected_resource.www_authenticate_value()


@pytest.mark.parametrize("path", ["/mcp", "/"])
def test_an_unauthenticated_mcp_call_carries_the_challenge(fresh_settings, path):
    """Against the REAL transport router, on both protocol paths — the
    challenge has to survive the router, the exception handler, and onto the
    wire. The dispatcher is never reached: the 401 is raised first."""
    fresh_settings(MCP_OAUTH_RESOURCE_URL="https://gw.example.test/api/mcp")

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(transport_mod.build_router(dispatcher=None))  # type: ignore[arg-type]

    resp = TestClient(app).post(path, json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})

    assert resp.status_code == 401
    assert resp.headers["WWW-Authenticate"] == (
        'Bearer resource_metadata="https://gw.example.test/api/mcp/.well-known/oauth-protected-resource"'
    )


def test_the_error_handler_forwards_headers(fresh_settings):
    """🚨 This repo installs its own StarletteHTTPException handler, replacing
    Starlette's — which forwards ``exc.headers``. It did not, so every
    WWW-Authenticate / Retry-After a raiser attached was silently dropped
    before the response left the process."""
    app = FastAPI()
    install_exception_handlers(app)
    probe = APIRouter()

    @probe.get("/boom")
    async def _boom():  # type: ignore[no-untyped-def]
        raise HTTPException(status_code=429, detail="slow down", headers={"Retry-After": "30"})

    app.include_router(probe)

    resp = TestClient(app).get("/boom")

    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "30"
