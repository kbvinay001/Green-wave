@echo off
:: Green Wave++ -- expose the local demo through a free Cloudflare quick
:: tunnel (no account needed). Run run.py first, then this; share the
:: printed https://....trycloudflare.com URL with /#key=<your api key>.

where cloudflared >nul 2>&1
if errorlevel 1 (
    echo cloudflared not found. Install it once with:
    echo     winget install Cloudflare.cloudflared
    exit /b 1
)

cloudflared tunnel --url http://localhost:8000
