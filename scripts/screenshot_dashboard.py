#!/usr/bin/env python3
"""
Grab a PNG of the live dashboard with headless Chrome.

Chrome's plain --screenshot flag shoots as soon as the page "loads",
and --virtual-time-budget fast-forwards timers in a way that starves
the token-fetch -> WebSocket chain -- either way you get a frozen
OFFLINE dashboard. So instead: launch headless Chrome with DevTools
open, let the page stream real telemetry for --wait seconds, then ask
the protocol for a capture of whatever is actually on screen.

Usage (backend must be running):
    python scripts/screenshot_dashboard.py docs/img/dashboard_live.png \
        --url "http://localhost:8000/#key=<api key>" --wait 5
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]


def find_chrome() -> str:
    for p in CHROME_CANDIDATES:
        if Path(p).exists():
            return p
    sys.exit("No Chrome/Edge found -- edit CHROME_CANDIDATES")


def page_ws_url(port: int, deadline_s: float = 15.0) -> str:
    """Poll DevTools until our page target shows up; return its WS URL."""
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list") as r:
                for target in json.load(r):
                    if target.get("type") == "page":
                        return target["webSocketDebuggerUrl"]
        except OSError:
            pass
        time.sleep(0.25)
    sys.exit("Chrome's DevTools endpoint never came up")


def main() -> None:
    ap = argparse.ArgumentParser(description="Screenshot the live dashboard")
    ap.add_argument("out", help="output PNG path")
    ap.add_argument("--url",  default="http://localhost:8000/")
    ap.add_argument("--wait", type=float, default=5.0,
                    help="seconds of live streaming before the capture")
    ap.add_argument("--width",  type=int, default=1500)
    ap.add_argument("--height", type=int, default=950)
    ap.add_argument("--port",   type=int, default=9777)
    args = ap.parse_args()

    chrome = find_chrome()
    profile = tempfile.mkdtemp(prefix="gw_shot_")
    proc = subprocess.Popen(
        [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--remote-debugging-port={args.port}",
         f"--user-data-dir={profile}",
         f"--window-size={args.width},{args.height}",
         args.url],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        ws_url = page_ws_url(args.port)
        with connect(ws_url, max_size=32 * 1024 * 1024) as ws:
            time.sleep(args.wait)              # real seconds of live frames
            ws.send(json.dumps({"id": 1, "method": "Page.captureScreenshot",
                                "params": {"format": "png"}}))
            while True:
                msg = json.loads(ws.recv(timeout=20))
                if msg.get("id") == 1:
                    break
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(msg["result"]["data"]))
        print(f"wrote {out}")
    finally:
        proc.kill()


if __name__ == "__main__":
    main()
