#!/usr/bin/env python3
"""
FastAPI WebSocket server -- Green Wave++

Endpoints:
  WS  /ws?token=  Real-time telemetry push to dashboard clients (~10 Hz).
                  Requires a live token from POST /token.
  GET /status     Health-check + uptime + client count. The one open door:
                  it leaks nothing but uptime, and docker/uptime probes
                  need somewhere unauthenticated to poke.
  POST /token     Trade the API key for a short-lived WS token.
  GET /beliefs    Current lane belief snapshot (API key required).
  POST /reset     Clear fusion state mid-demo (API key required, audited).
                  Used to be GET -- it mutates state, so it's POST now.

Auth model (phase 4): every request except /status carries X-API-Key.
Browsers can't set headers on WebSockets, so the dashboard first POSTs
/token and connects with ?token=...; tokens expire after
security.ws_token_ttl_sec and live only in this process's memory.

The pipeline is attached via attach_pipeline() from run.py after both
the pipeline and server are initialised.  This avoids circular imports
and lets the server start independently for testing.
"""

from __future__ import annotations

import asyncio
import json
import secrets as pysecrets
import time
from pathlib import Path
from typing import Optional, Set

import uvicorn
import yaml
from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

from common.security import WSTokenStore, resolve_api_key

ROOT = Path(__file__).resolve().parents[2]


def _load_security_config() -> dict:
    try:
        with open(ROOT / "common" / "config.yaml") as f:
            return (yaml.safe_load(f) or {}).get("security", {}) or {}
    except FileNotFoundError:
        return {}


_sec_cfg              = _load_security_config()
_api_key, _key_source = resolve_api_key(_sec_cfg)
_tokens               = WSTokenStore(ttl_sec=float(_sec_cfg.get("ws_token_ttl_sec", 300)))


def configure_security(api_key: Optional[str] = None,
                       ws_token_ttl_sec: Optional[float] = None) -> None:
    """Swap the live key/token store -- tests use this; run.py never needs to."""
    global _api_key, _tokens
    if api_key is not None:
        _api_key = api_key
    if ws_token_ttl_sec is not None:
        _tokens = WSTokenStore(ttl_sec=ws_token_ttl_sec)


app = FastAPI(title="GreenWave++", version="1.0.0", docs_url="/docs")

# Exact origins from config -- the old allow_origins=["*"] let any web page
# a browser wandered onto subscribe to the telemetry stream.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = list(_sec_cfg.get("cors_origins")
                             or ["http://localhost:5173", "http://127.0.0.1:5173"]),
    allow_credentials = False,
    allow_methods     = ["GET", "POST", "OPTIONS"],
    allow_headers     = ["X-API-Key", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def require_api_key(x_api_key: str = Header(default="")) -> None:
    if not pysecrets.compare_digest(x_api_key, _api_key):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TokenRequest(BaseModel):
    # never longer than an hour, no matter how politely a client asks
    ttl_sec: Optional[float] = Field(default=None, gt=0, le=3600)


class TokenResponse(BaseModel):
    token:       str
    expires_at:  float
    ttl_sec:     float


class StatusResponse(BaseModel):
    status:            str
    uptime_sec:        float
    connected_clients: int
    pipeline_running:  bool


class BeliefsResponse(BaseModel):
    beliefs: dict
    phases:  dict


class MessageResponse(BaseModel):
    status:  str
    message: str


class WSClientMessage(BaseModel):
    """The only thing a dashboard may send upstream. Everything else is dropped."""
    type: str = Field(pattern="^ping$")


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_clients:      Set[WebSocket] = set()
_pipeline                     = None     # injected by attach_pipeline()
_server_start: float          = time.time()


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket, token: str = Query(default="")) -> None:
    if not _tokens.validate(token):
        # Reject before the handshake completes -- the client sees 403.
        await ws.close(code=1008)
        return

    await ws.accept()
    _clients.add(ws)
    print(f"[GREEN] Dashboard connected  ({len(_clients)} active)")

    try:
        # Keep the socket alive; data flows out via broadcast(). A silent
        # client is fine (the timeout just re-arms), a malformed inbound
        # message is dropped, and a valid ping gets a pong.
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                continue
            try:
                msg = WSClientMessage.model_validate_json(raw)
            except ValidationError:
                continue
            if msg.type == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _clients.discard(ws)
        print(f"[RED] Dashboard disconnected  ({len(_clients)} active)")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    return StatusResponse(
        status            = "ok",
        uptime_sec        = round(time.time() - _server_start, 1),
        connected_clients = len(_clients),
        pipeline_running  = _pipeline is not None,
    )


@app.post("/token", response_model=TokenResponse,
          dependencies=[Depends(require_api_key)])
async def token(req: Optional[TokenRequest] = None) -> TokenResponse:
    ttl = req.ttl_sec if (req and req.ttl_sec) else None
    tok, expires = _tokens.issue(ttl_sec=ttl)
    return TokenResponse(token=tok, expires_at=expires,
                         ttl_sec=ttl if ttl else _tokens.ttl)


@app.post("/reset", response_model=MessageResponse,
          dependencies=[Depends(require_api_key)])
async def reset() -> MessageResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="No pipeline attached")
    _pipeline.fusion.reset_all()
    audit = getattr(_pipeline, "audit", None)
    if audit is not None:
        audit.append("fusion_reset", {"via": "api"})
    return MessageResponse(status="ok", message="Fusion state cleared")


@app.get("/beliefs", response_model=BeliefsResponse,
         dependencies=[Depends(require_api_key)])
async def beliefs() -> BeliefsResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="No pipeline attached")
    return BeliefsResponse(
        beliefs = _pipeline.fusion.get_beliefs(),
        phases  = _pipeline.fusion.get_phases(),
    )


# ---------------------------------------------------------------------------
# Broadcast helper
# ---------------------------------------------------------------------------

async def broadcast(data: dict) -> None:
    """Push a telemetry frame to all connected dashboard clients."""
    if not _clients:
        return

    msg  = json.dumps(data)
    dead: Set[WebSocket] = set()

    for ws in list(_clients):
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)

    # Use difference_update (in-place method) instead of -= so Python doesn't
    # treat _clients as a local variable due to augmented-assignment scoping.
    _clients.difference_update(dead)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def attach_pipeline(pipeline) -> None:
    """
    Called by run.py after pipeline initialisation.
    Injects broadcast into the pipeline so it can push updates.
    """
    global _pipeline
    _pipeline = pipeline
    pipeline.set_broadcast(broadcast)
    print("[OK] Pipeline attached to WebSocket server")


# ---------------------------------------------------------------------------
# Standalone entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("ui.backend.server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
