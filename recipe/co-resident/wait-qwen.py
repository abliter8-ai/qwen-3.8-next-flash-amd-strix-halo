#!/usr/bin/env python3
"""Keep Bonsai behind Qwen's high-memory startup."""
import json
import os
import time
import urllib.request

url = os.environ.get("QWEN_BASE_URL", "http://127.0.0.1:8731").rstrip("/") + "/health"
deadline = time.monotonic() + 500
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            state = json.load(response)
        if state.get("status") == "ok" and state.get("engine", {}).get("responds"):
            break
    except (OSError, ValueError):
        pass
    time.sleep(5)
else:
    raise SystemExit("Qwen is not ready; Bonsai stays down")
