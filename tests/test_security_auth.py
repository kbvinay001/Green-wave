"""Phase 4 auth: API key resolution, locked REST endpoints, WS tokens, CORS."""
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.audit import AuditLog, verify          # noqa: E402
from common.security import WSTokenStore, resolve_api_key  # noqa: E402
import ui.backend.server as srv                    # noqa: E402

try:
    from starlette.testclient import WebSocketDenialResponse
    DENIED = (WebSocketDisconnect, WebSocketDenialResponse)
except ImportError:           # older starlette closes instead of denying
    DENIED = (WebSocketDisconnect,)

KEY = "test-key-123"


class _FakeFusion:
    def __init__(self):
        self.reset_calls = 0
    def reset_all(self):
        self.reset_calls += 1
    def get_beliefs(self):
        return {"approach_west": 0.42}
    def get_phases(self):
        return {"approach_west": "IDLE"}


class _FakePipeline:
    def __init__(self, audit=None):
        self.fusion = _FakeFusion()
        self.audit  = audit
    def set_broadcast(self, fn):
        pass


@pytest.fixture()
def client():
    srv.configure_security(api_key=KEY, ws_token_ttl_sec=300)
    srv._pipeline = None
    with TestClient(srv.app) as c:
        yield c
    srv._pipeline = None


# ------------------------------------------------------------------
# API key resolution order
# ------------------------------------------------------------------

def test_key_from_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("GREENWAVE_API_KEY", "from-the-env")
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "secrets.yaml").write_text("api_key: from-the-file\n")
    key, source = resolve_api_key({}, root=tmp_path)
    assert (key, source) == ("from-the-env", "env")


def test_key_from_secrets_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GREENWAVE_API_KEY", raising=False)
    (tmp_path / "common").mkdir()
    (tmp_path / "common" / "secrets.yaml").write_text("api_key: from-the-file\n")
    key, source = resolve_api_key({}, root=tmp_path)
    assert (key, source) == ("from-the-file", "file")


def test_key_generated_once_then_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("GREENWAVE_API_KEY", raising=False)
    key1, source1 = resolve_api_key({}, root=tmp_path)
    assert source1 == "generated" and len(key1) > 30
    # second resolve finds the file the first one wrote
    key2, source2 = resolve_api_key({}, root=tmp_path)
    assert (key2, source2) == (key1, "file")


# ------------------------------------------------------------------
# REST auth
# ------------------------------------------------------------------

def test_status_is_open(client):
    r = client.get("/status")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_beliefs_needs_key(client):
    assert client.get("/beliefs").status_code == 401
    assert client.get("/beliefs", headers={"X-API-Key": "wrong"}).status_code == 401

    srv._pipeline = _FakePipeline()
    r = client.get("/beliefs", headers={"X-API-Key": KEY})
    assert r.status_code == 200
    assert r.json()["beliefs"] == {"approach_west": 0.42}


def test_reset_needs_key_and_is_audited(client, tmp_path):
    assert client.post("/reset").status_code == 401

    audit_path = tmp_path / "audit.jsonl"
    pipe = _FakePipeline(audit=AuditLog(audit_path))
    srv._pipeline = pipe

    r = client.post("/reset", headers={"X-API-Key": KEY})
    assert r.status_code == 200
    assert pipe.fusion.reset_calls == 1

    ok, _ = verify(audit_path)
    assert ok
    assert "fusion_reset" in audit_path.read_text()


def test_old_get_reset_route_is_gone(client):
    # mutating state over GET was the phase-3 wart this phase fixes.
    # 405 without the SPA mount, 404 falling through to it -- either way,
    # nothing mutates.
    assert client.get("/reset", headers={"X-API-Key": KEY}).status_code in (404, 405)


# ------------------------------------------------------------------
# WS tokens
# ------------------------------------------------------------------

def test_token_endpoint_needs_key(client):
    assert client.post("/token").status_code == 401
    r = client.post("/token", headers={"X-API-Key": KEY})
    assert r.status_code == 200
    body = r.json()
    assert body["token"] and body["expires_at"] > time.time()


def test_token_ttl_is_bounded(client):
    headers = {"X-API-Key": KEY}
    assert client.post("/token", json={"ttl_sec": -5},    headers=headers).status_code == 422
    assert client.post("/token", json={"ttl_sec": 99999}, headers=headers).status_code == 422
    assert client.post("/token", json={"ttl_sec": 60},    headers=headers).status_code == 200


def test_ws_rejects_missing_or_bogus_token(client):
    for url in ("/ws", "/ws?token=bogus"):
        with pytest.raises(DENIED):
            with client.websocket_connect(url):
                pass


def test_ws_accepts_live_token_and_validates_messages(client):
    tok = client.post("/token", headers={"X-API-Key": KEY}).json()["token"]
    with client.websocket_connect(f"/ws?token={tok}") as ws:
        ws.send_text("{not json at all")          # dropped silently
        ws.send_text('{"type": "selfdestruct"}')  # schema says no
        ws.send_text('{"type": "ping"}')          # the one valid message
        assert ws.receive_json() == {"type": "pong"}


def test_ws_rejects_expired_token(client):
    srv.configure_security(api_key=KEY, ws_token_ttl_sec=0.05)
    tok = client.post("/token", headers={"X-API-Key": KEY}).json()["token"]
    time.sleep(0.1)
    with pytest.raises(DENIED):
        with client.websocket_connect(f"/ws?token={tok}"):
            pass


def test_token_store_expiry_directly():
    t = [1000.0]
    store = WSTokenStore(ttl_sec=10, now=lambda: t[0])
    tok, expires = store.issue()
    assert store.validate(tok) and expires == 1010.0
    t[0] = 1011.0
    assert not store.validate(tok)
    assert not store.validate("never-issued")


# ------------------------------------------------------------------
# CORS
# ------------------------------------------------------------------

def test_cors_allows_only_configured_origins(client):
    good = client.get("/status", headers={"Origin": "http://localhost:5173"})
    assert good.headers.get("access-control-allow-origin") == "http://localhost:5173"

    evil = client.get("/status", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in evil.headers
