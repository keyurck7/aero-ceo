"""Serve any AERO-CEO MCP server over stateless Streamable HTTP.

Adds what a container platform needs around the MCP endpoint:
``GET /healthz`` (liveness and readiness), DNS-rebinding protection with an explicit
host allow-list, and an interim bearer-token check. Real OAuth 2.1 sits at the
MCP gateway in roadmap phase 6.
"""

from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from aero import __version__
from aero.config import Settings


class BearerAuth:
    """ASGI middleware: require ``Authorization: Bearer <token>`` except on /healthz."""

    def __init__(self, app: ASGIApp, token: str, open_paths: tuple[str, ...] = ("/healthz",)) -> None:
        self.app, self.token, self.open_paths = app, token.encode(), open_paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path") not in self.open_paths:
            header = dict(scope.get("headers") or []).get(b"authorization", b"")
            supplied = header[7:] if header[:7].lower() == b"bearer " else b""
            if not hmac.compare_digest(supplied, self.token):
                await JSONResponse({"error": "unauthorized"}, status_code=401,
                                   headers={"WWW-Authenticate": "Bearer"})(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_http_app(server: MCPServer, settings: Settings, ready: Callable[[], dict[str, Any]] | None = None) -> ASGIApp:
    hosts = settings.mcp_allowed_host_list
    security = (
        TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts,
                                  allowed_origins=[f"http://{h}" for h in hosts] + [f"https://{h}" for h in hosts])
        if hosts else None
    )
    app: Starlette = server.streamable_http_app(
        streamable_http_path=settings.mcp_path, stateless_http=True, json_response=True,
        transport_security=security, host=settings.mcp_host,
    )

    async def healthz(_: Request) -> Response:
        body: dict[str, Any] = {"status": "ok", "server": server.name, "version": __version__}
        if ready is not None:
            try:
                body["checks"] = ready()
            except Exception as exc:  # readiness failure is reported, never raised
                return JSONResponse({"status": "unavailable", "error": str(exc)}, status_code=503)
        return JSONResponse(body)

    app.router.routes.append(Route("/healthz", healthz, methods=["GET"]))
    return BearerAuth(app, settings.mcp_bearer_token) if settings.mcp_bearer_token else app
