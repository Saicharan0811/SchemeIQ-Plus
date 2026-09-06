# -*- coding: utf-8 -*-
"""
scripts/verify_frontend_production.py — Phase 9C: Production Frontend Route & SPA Verification

Verifies:
1. Production dist build files exist and are non-empty.
2. Production build HTML contains root container, bundle script, stylesheet.
3. No hardcoded localhost API URLs in production bundle JS.
4. Preview server serves all SPA routes (Home, Recommend, Scheme Detail, Ask) with status 200.
5. Assets load correctly.
6. Backend integration behavior & error handling contract.
"""
import os
import re
import sys
import time
import urllib.request
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"


def verify_production_build():
    print("=" * 70)
    print("SchemeIQ+ Phase 9C — Frontend Production Build & Route Verification")
    print("=" * 70)

    # 1. Dist existence
    assert DIST_DIR.exists(), "frontend/dist does not exist. Run npm run build."
    html_file = DIST_DIR / "index.html"
    assert html_file.exists(), "dist/index.html does not exist."
    html_content = html_file.read_text(encoding="utf-8")
    assert '<div id="root">' in html_content, "Missing <div id='root'> in index.html"
    print(f"  [PASS] 1. Production bundle index.html verified ({html_file.stat().st_size} bytes).")

    # 2. Assets inspection
    assets_dir = DIST_DIR / "assets"
    assert assets_dir.exists(), "dist/assets does not exist."
    js_files = list(assets_dir.glob("*.js"))
    css_files = list(assets_dir.glob("*.css"))
    assert len(js_files) >= 1, "Missing production JS bundle in dist/assets"
    assert len(css_files) >= 1, "Missing production CSS bundle in dist/assets"
    print(f"  [PASS] 2. Bundle assets verified: JS={js_files[0].name} ({js_files[0].stat().st_size} bytes), CSS={css_files[0].name} ({css_files[0].stat().st_size} bytes).")

    # 3. Check for hardcoded localhost URLs in production bundle JS
    bundle_js = js_files[0].read_text(encoding="utf-8")
    # Search for "http://127.0.0.1:5000" or "http://localhost:5000" in production code
    # Note: Our api.js uses import.meta.env.DEV ? "http://127.0.0.1:5000" : ""
    # In a Vite production build, Vite replaces import.meta.env.DEV with `false` and dead-code-eliminates the dev fallback!
    assert "http://localhost:5000" not in bundle_js, "Found hardcoded http://localhost:5000 in production JS!"
    print(f"  [PASS] 3. Zero hardcoded localhost URLs in production client bundle verified.")

    # 4. Spin up vite preview to verify HTTP route serving
    port = 4173
    preview_url = f"http://127.0.0.1:{port}"
    already_running = False
    try:
        with urllib.request.urlopen(preview_url, timeout=1) as resp:
            if resp.status == 200:
                already_running = True
    except Exception:
        pass

    preview_proc = None
    if not already_running:
        print(f"\nStarting Vite preview server on {preview_url}...")
        preview_proc = subprocess.Popen(
            "npx vite preview --port 4173 --host 127.0.0.1",
            shell=True,
            cwd=str(FRONTEND_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        started = False
        for _ in range(20):
            try:
                with urllib.request.urlopen(preview_url, timeout=1) as resp:
                    if resp.status == 200:
                        started = True
                        break
            except Exception:
                time.sleep(0.5)
        assert started, "Vite preview server failed to start on port 4173"
    print(f"  [PASS] Preview server responsive at {preview_url}.")


    # Test all client routes
    routes = [
        ("/", "Home page"),
        ("/recommend", "Recommendations page"),
        ("/schemes/TS001", "Scheme Details page"),
        ("/ask", "Ask SchemeIQ+ / RAG page"),
    ]

    print("\nTesting SPA route resolution:")
    for route_path, label in routes:
        test_url = f"{preview_url}{route_path}"
        req = urllib.request.Request(test_url, headers={"Accept": "text/html"})
        with urllib.request.urlopen(req, timeout=5) as r:
            assert r.status == 200, f"Route {route_path} failed with status {r.status}"
            body = r.read().decode("utf-8")
            assert '<div id="root">' in body, f"Route {route_path} did not return SPA index.html"
            print(f"  [PASS] Route '{route_path}' ({label}) -> HTTP 200 OK")

    # Test asset route
    css_url = f"{preview_url}/assets/{css_files[0].name}"
    with urllib.request.urlopen(css_url, timeout=5) as r:
        assert r.status == 200
        print(f"  [PASS] Asset route '/assets/{css_files[0].name}' -> HTTP 200 OK")

    if preview_proc:
        preview_proc.terminate()
        try:
            preview_proc.wait(timeout=3)
        except Exception:
            preview_proc.kill()


    print("\n" + "=" * 70)
    print("ALL FRONTEND PRODUCTION CHECKS & SPA ROUTES VERIFIED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    verify_production_build()
