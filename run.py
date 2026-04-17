#!/usr/bin/env python3
"""
Green Wave++ -- unified launcher

Usage:
    python run.py --demo          # synthetic data, no hardware needed
    python run.py                 # live mic + camera (stubs to be wired)
    python run.py --no-ui         # backend only, skip Vite
    python run.py --port 9000     # custom port

What this does:
  1. Loads config.yaml
  2. Builds EndToEndPipeline (demo mode if --demo)
  3. Attaches the pipeline to the FastAPI WebSocket server
  4. Starts Vite dev server for the React dashboard (if node_modules exists)
  5. Opens the browser to the dashboard URL
  6. Runs uvicorn + fusion loop concurrently until Ctrl-C
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import uvicorn
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Green Wave++ launcher")
    ap.add_argument("--demo",   action="store_true",
                    help="Run with synthetic data -- no hardware or trained models needed")
    ap.add_argument("--port",   type=int, default=8000, help="Backend WebSocket port (default 8000)")
    ap.add_argument("--no-ui",  action="store_true",
                    help="Start backend only; skip the Vite dev server")
    return ap.parse_args()


def load_config() -> dict:
    cfg_path = ROOT / "common" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------

async def run_all(pipeline, port: int) -> None:
    """Start FastAPI server and the fusion loop as concurrent async tasks."""
    import ui.backend.server as srv
    srv.attach_pipeline(pipeline)

    server = uvicorn.Server(
        uvicorn.Config(
            "ui.backend.server:app",
            host      = "0.0.0.0",
            port      = port,
            log_level = "warning",
        )
    )

    pipeline.start()

    try:
        await asyncio.gather(
            server.serve(),
            pipeline.run_fusion_loop(),
        )
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------

def main() -> None:
    args   = parse_args()
    config = load_config()

    from integration.pipeline import EndToEndPipeline
    pipeline = EndToEndPipeline(config, demo=args.demo)

    # Optionally start the Vite dev server
    vite_proc = None
    if not args.no_ui:
        ui_dir = ROOT / "ui"
        if (ui_dir / "node_modules").exists():
            vite_proc = subprocess.Popen(
                ["npm.cmd", "run", "dev", "--", "--port", "5173"],
                cwd     = str(ui_dir),
                stdout  = subprocess.DEVNULL,
                stderr  = subprocess.DEVNULL,
            )
            time.sleep(2.5)
            webbrowser.open("http://localhost:5173")
            print("[WEB] Dashboard -> http://localhost:5173")
        else:
            print("[WARN] ui/node_modules missing -- run: cd greenwave/ui && npm install")

    mode = "DEMO (synthetic)" if args.demo else "LIVE"
    print(f"[>>] Green Wave++  |  backend -> http://localhost:{args.port}  |  mode={mode}")
    print("   Press Ctrl-C to stop\n")

    try:
        asyncio.run(run_all(pipeline, args.port))
    except KeyboardInterrupt:
        print("\n[STOP] Shutting down...")
    finally:
        pipeline.stop()
        if vite_proc:
            vite_proc.terminate()


if __name__ == "__main__":
    main()
