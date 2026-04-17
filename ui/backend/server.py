#!/usr/bin/env python3
"""
FastAPI WebSocket server -- Green Wave++

Endpoints:
  WS  /ws        Real-time telemetry push to dashboard clients (~10 Hz)
  GET /status    Health-check + uptime + connected client count
  GET /reset     Clear fusion state (useful mid-demo)
  GET /beliefs   Current lane belief snapshot (REST, no WS subscription needed)

The pipeline is attached via attach_pipeline() from run.py after both
the pipeline and server are initialised.  This avoids circular imports
and lets the server start independently for testing.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="GreenWave++", version="1.0.0", docs_url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _clients.add(ws)
    print(f"[GREEN] Dashboard connected  ({len(_clients)} active)")

    try:
        # Block here, keeping the socket alive.  Data is pushed via broadcast().
        # We use a long timeout on receive so the loop doesn't spin.
        while True:
            await asyncio.wait_for(ws.receive_text(), timeout=30.0)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception as e:
        pass
    finally:
        _clients.discard(ws)
        print(f"[RED] Dashboard disconnected  ({len(_clients)} active)")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/status")
async def status() -> JSONResponse:
    return JSONResponse({
        "status":             "ok",
        "uptime_sec":         round(time.time() - _server_start, 1),
        "connected_clients":  len(_clients),
        "pipeline_running":   _pipeline is not None,
    })


@app.get("/reset")
async def reset() -> JSONResponse:
    if _pipeline is None:
        return JSONResponse({"status": "error", "message": "No pipeline attached"}, status_code=503)
    _pipeline.fusion.reset_all()
    return JSONResponse({"status": "ok", "message": "Fusion state cleared"})


@app.get("/beliefs")
async def beliefs() -> JSONResponse:
    if _pipeline is None:
        return JSONResponse({"status": "error"}, status_code=503)
    return JSONResponse({
        "beliefs": _pipeline.fusion.get_beliefs(),
        "phases":  _pipeline.fusion.get_phases(),
    })


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
