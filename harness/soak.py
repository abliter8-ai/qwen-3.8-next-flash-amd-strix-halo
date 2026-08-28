#!/usr/bin/env python3
"""IP-276 Task 13 mixed-capability soak — minimum 30 minutes, one server process.

Sequence (sequential, --parallel 1 is locked):
  10 multi-turn text conversations, 10 reasoning-on, 10 reasoning-off,
  10 forced tool calls, 10 image requests (fixed local fixture),
  5 deterministic repeat-pairs (same prompt twice, hashes recorded),
  1 request with >=48K input tokens, then continuous free-form generation
  until >=30 minutes total elapsed.
Health + residency sampled every 30 s into OUTPUT_DIR/samples.jsonl.
Requests logged to OUTPUT_DIR/requests.jsonl.
"""
import base64
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

PLAN_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://localhost:8080")
MODEL = os.environ.get("LLAMA_MODEL", "qwen3.8-flash-next")
# any local JPEG works; set VISION_FIXTURE to point at yours
IMAGE = Path(os.environ.get("VISION_FIXTURE", PLAN_ROOT / "fixtures/vision.jpg"))
MIN_SECONDS = 30 * 60

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
REQ_LOG = (OUT / "requests.jsonl").open("a")
stop_sampler = threading.Event()


def sampler():
    pid = subprocess.run(["pgrep", "-n", "-x", "llama-server"], capture_output=True, text=True).stdout.strip()
    with (OUT / "samples.jsonl").open("a") as f:
        while not stop_sampler.is_set():
            s = {"t": time.time()}
            try:
                t0 = time.monotonic()
                with urllib.request.urlopen(f"{BASE_URL}/health", timeout=10) as r:
                    s["health"] = json.loads(r.read().decode()).get("status")
                s["health_ms"] = round((time.monotonic() - t0) * 1000, 1)
            except Exception as e:
                s["health"] = f"ERR:{type(e).__name__}"
            try:
                status = Path(f"/proc/{pid}/status").read_text()
                for k in ("VmRSS", "RssAnon", "RssFile"):
                    s[k] = int([l for l in status.splitlines() if l.startswith(k)][0].split()[1])
                s["pid"] = pid
            except Exception:
                s["proc"] = "gone"
            try:
                mem = Path("/proc/meminfo").read_text()
                s["MemAvailable"] = int([l for l in mem.splitlines() if l.startswith("MemAvailable")][0].split()[1])
                s["gtt"] = int(Path("/sys/class/drm/card0/device/mem_info_gtt_used").read_text())
            except Exception:
                pass
            f.write(json.dumps(s) + "\n")
            f.flush()
            stop_sampler.wait(30)


def chat(name, messages, **extra):
    payload = {"model": MODEL, "messages": messages, "temperature": 0,
               "max_tokens": 256, "stream": False}
    payload.update(extra)
    req = urllib.request.Request(f"{BASE_URL}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    ok, content, finish, err = False, "", None, None
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.loads(r.read().decode())
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        if not content and msg.get("tool_calls"):
            content = json.dumps(msg["tool_calls"])
        finish = resp["choices"][0].get("finish_reason")
        ok = bool(content.strip()) and finish in ("stop", "length", "tool_calls")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    rec = {"name": name, "ok": ok, "finish": finish, "err": err,
           "sha": hashlib.sha256(content.encode()).hexdigest()[:16],
           "secs": round(time.monotonic() - t0, 1), "t": time.time()}
    REQ_LOG.write(json.dumps(rec) + "\n")
    REQ_LOG.flush()
    print(rec["name"], "ok" if ok else f"FAIL {err}", rec["secs"], "s", flush=True)
    return rec


def main():
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t_start = time.monotonic()
    fails = 0

    for i in range(10):
        r = chat(f"multiturn-{i}", [
            {"role": "user", "content": f"My reference code is X{i}77. Confirm it."},
            {"role": "assistant", "content": "Confirmed."},
            {"role": "user", "content": "Repeat my reference code exactly."}])
        fails += 0 if r["ok"] else 1
    for i in range(10):
        # thinking-on needs headroom to close the think block: measured floor
        # ~544 tokens on this lane; a 256 cap yields finish=length with zero
        # visible content (verifier D3). Caller contract: thinking-on => >=1024.
        r = chat(f"reason-on-{i}", [{"role": "user", "content": f"Compute {13+i} * {29+i} step by step."}],
                 chat_template_kwargs={"enable_thinking": True}, max_tokens=1024)
        fails += 0 if r["ok"] else 1
    for i in range(10):
        r = chat(f"reason-off-{i}", [{"role": "user", "content": f"What is {211+i} + {388+i}? Number only."}],
                 chat_template_kwargs={"enable_thinking": False})
        fails += 0 if r["ok"] else 1
    tool = {"type": "function", "function": {"name": "lookup_weather",
            "description": "Weather for a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                           "required": ["city"], "additionalProperties": False}}}
    cities = ["Dublin", "Cork", "Galway", "Limerick", "Sligo", "Derry", "Belfast", "Waterford", "Kilkenny", "Athlone"]
    for i in range(10):
        r = chat(f"tool-{i}", [{"role": "user", "content": f"Use the tool for weather in {cities[i]}."}],
                 tools=[tool], tool_choice={"type": "function", "function": {"name": "lookup_weather"}})
        fails += 0 if r["ok"] else 1
    img64 = base64.b64encode(IMAGE.read_bytes()).decode()
    for i in range(10):
        # vision callers at short caps must disable thinking (verifier D3):
        # default xhigh thinking burns a 256 cap inside <think> -> empty reply.
        r = chat(f"vision-{i}", [{"role": "user", "content": [
            {"type": "text", "text": f"Question {i}: what colours dominate this image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img64}"}}]}],
            chat_template_kwargs={"enable_thinking": False})
        fails += 0 if r["ok"] else 1
    for i in range(5):
        a = chat(f"detpair-{i}-a", [{"role": "user", "content": f"List the first {5+i} prime numbers."}],
                 chat_template_kwargs={"enable_thinking": False})
        b = chat(f"detpair-{i}-b", [{"role": "user", "content": f"List the first {5+i} prime numbers."}],
                 chat_template_kwargs={"enable_thinking": False})
        fails += 0 if (a["ok"] and b["ok"]) else 1
        REQ_LOG.write(json.dumps({"name": f"detpair-{i}-equal", "equal": a["sha"] == b["sha"]}) + "\n")

    para = ("Watch entry %d: course steady, fuel nominal, reefers in range, "
            "vibration within tolerance, next check in thirty minutes. ")
    big = "".join(para % i for i in range(3400)) + "\nHow many watch entries are listed? Number only."
    r = chat("deep-48k", [{"role": "user", "content": big}], max_tokens=64)
    fails += 0 if r["ok"] else 1

    i = 0
    while time.monotonic() - t_start < MIN_SECONDS:
        r = chat(f"freeform-{i}", [{"role": "user", "content":
                 f"Continue this saga, part {i}: the fleet crossed the meridian and"}],
                 max_tokens=512, chat_template_kwargs={"enable_thinking": False})
        fails += 0 if r["ok"] else 1
        i += 1

    elapsed = round(time.monotonic() - t_start, 1)
    stop_sampler.set()
    th.join(timeout=5)
    summary = {"elapsed_seconds": elapsed, "failures": fails, "freeform_rounds": i}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
