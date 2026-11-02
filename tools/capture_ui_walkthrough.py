"""
Phase R9 Step R9.3: Automated Streamlit Studio UI Walkthrough Screenshot Tool
Controls headless Chrome via Chrome DevTools Protocol (CDP) to navigate through
all Streamlit Studio tabs and capture high-resolution viewport screenshots.
"""
import os
import sys
import io

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import time
import base64
import asyncio
import subprocess
import urllib.request
from pathlib import Path
import websockets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "docs_generated" / "v2" / "ui_walkthrough"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
STREAMLIT_URL = "http://localhost:8501/"
DEBUG_PORT = 9222


async def run_walkthrough():
    print("=" * 80)
    print("  PHASE R9: STREAMLIT STUDIO UI AUTOMATION & CAPTURE")
    print("=" * 80)

    # Launch Chrome with remote debugging
    chrome_cmd = [
        CHROME_PATH,
        "--headless=new",
        f"--remote-debugging-port={DEBUG_PORT}",
        "--remote-allow-origins=*",
        "--disable-gpu",
        "--no-sandbox",
        "--window-size=1440,1100",
        STREAMLIT_URL
    ]
    print(f"Launching Chrome Headless on port {DEBUG_PORT}...")
    chrome_proc = subprocess.Popen(chrome_cmd)

    ws_url = None
    target_page = None
    for attempt in range(15):
        time.sleep(1.0)
        try:
            req = urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json", timeout=2)
            pages = json.loads(req.read().decode("utf-8"))
            for p in pages:
                if p.get("type") == "page" and "8501" in p.get("url", ""):
                    target_page = p
                    ws_url = p["webSocketDebuggerUrl"].replace("localhost", "127.0.0.1")
                    break
            if ws_url:
                print(f"Connected to Streamlit page: '{target_page.get('title')}'")
                print(f"WebSocket URL: {ws_url}")
                break
        except Exception:
            pass

    if not ws_url:
        print("ERROR: Failed to locate Streamlit target page in Chrome.")
        chrome_proc.terminate()
        return

    try:
        async with websockets.connect(ws_url, max_size=50_000_000) as ws:
            mid = 1
            async def call(method, params=None):
                nonlocal mid
                cur_id = mid
                mid += 1
                await ws.send(json.dumps({"id": cur_id, "method": method, "params": params or {}}))
                while True:
                    resp = await asyncio.wait_for(ws.recv(), timeout=15)
                    d = json.loads(resp)
                    if d.get("id") == cur_id:
                        return d

            async def eval_js(expr):
                r = await call("Runtime.evaluate", {"expression": expr, "returnByValue": True})
                return r.get("result", {}).get("result", {}).get("value")

            # Enable Page
            await call("Page.enable")

            # Poll for Streamlit tabs hydration (up to 15s)
            print("Waiting for Streamlit app hydration...")
            tabs_ready = False
            for poll_idx in range(15):
                n_tabs = await eval_js('document.querySelectorAll("button[role=\\"tab\\"]").length')
                if n_tabs and n_tabs >= 4:
                    tabs_ready = True
                    print(f"Streamlit hydrated! Detected {n_tabs} active tabs.")
                    break
                await asyncio.sleep(1.0)

            tab_files = [
                "01_tab1_storyboard_explorer.png",
                "02_tab2_visual_dna_anchors.png",
                "03_tab3_diagnostics_benchmark.png",
                "04_tab4_frozen_verifier.png"
            ]

            tab_labels = await eval_js('Array.from(document.querySelectorAll("button[role=\\"tab\\"]")).map(b => b.innerText.trim())')
            print(f"Tab labels found: {tab_labels}")

            for idx in range(min(len(tab_files), len(tab_labels) if tab_labels else 4)):
                print(f"\n[TAB {idx + 1}/4] Navigating to: {tab_labels[idx] if tab_labels and idx < len(tab_labels) else idx}...")
                click_js = f'document.querySelectorAll("button[role=\\"tab\\"]")[{idx}].click();'
                await eval_js(click_js)
                await asyncio.sleep(3.0)

                # Capture screenshot
                shot = await call("Page.captureScreenshot", {"format": "png"})
                b64 = shot.get("result", {}).get("data")
                if b64:
                    out_path = OUTPUT_DIR / tab_files[idx]
                    raw_bytes = base64.b64decode(b64)
                    with open(out_path, "wb") as f:
                        f.write(raw_bytes)
                    print(f"  ✅ Saved screenshot: {out_path.name} ({len(raw_bytes)} bytes)")

            print("\n🎉 All 4 UI walkthrough screenshots successfully captured!")

    finally:
        chrome_proc.terminate()
        chrome_proc.wait()
        print("Chrome closed.")


def main():
    asyncio.run(run_walkthrough())


if __name__ == "__main__":
    main()
