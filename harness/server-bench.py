#!/usr/bin/env python3
"""IP-276 server-mode fixed-prompt benchmark.

Three fixed prompts (prose / code / structured), one discarded warm-up plus
five retained runs each, temperature 0, thinking off (identical across every
comparison arm). Captures llama-server timings, wall clock, token counts,
finish reason, and an output hash per run.

Usage: server-bench.py OUTPUT_JSON
"""
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://localhost:8080")
MODEL = os.environ.get("LLAMA_MODEL", "qwen3.8-flash-next")
RUNS = 5

PROMPTS = {
    "prose": (
        "Write a detailed, flowing essay (no headings, no lists) on the history "
        "of container shipping: origins with Malcom McLean, the standardisation "
        "of the TEU, the effect on port labour, the rise of mega-ships, and how "
        "containerisation reshaped global supply chains. Aim for roughly 900 "
        "words of continuous prose."
    ),
    "code": (
        "Write a complete, production-quality Python module implementing a "
        "token-bucket rate limiter with these requirements: thread-safe, "
        "monotonic-clock based, supports per-key buckets with an LRU cap of "
        "10000 keys, a decorator interface, a context-manager interface, full "
        "type hints, docstrings, and a small doctest section. Output only code."
    ),
    "structured": (
        "Produce a JSON document (and nothing else) describing a fictional "
        "fleet of 12 cargo ships. Each ship object must contain: name (string), "
        "imo_number (9-digit string), teu_capacity (integer), year_built "
        "(integer), flag_state (string), current_port (string), heading_degrees "
        "(number), fuel (object with type and tonnes_remaining), and a "
        "maintenance_log array of exactly 3 entries each with date, yard, and "
        "work_done fields. Use realistic values."
    ),
}


def one_run(name, prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1000,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "timings_per_token": False,
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.loads(r.read().decode())
    wall = time.monotonic() - t0
    content = resp["choices"][0]["message"].get("content") or ""
    timings = resp.get("timings", {})
    usage = resp.get("usage", {})
    return {
        "prompt_name": name,
        "wall_seconds": round(wall, 3),
        "finish_reason": resp["choices"][0].get("finish_reason"),
        "usage": usage,
        "timings": timings,
        "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "output_chars": len(content),
    }


def summarise(runs, key_fn):
    vals = [key_fn(r) for r in runs if key_fn(r) is not None]
    if not vals:
        return None
    med = statistics.median(vals)
    return {
        "median": round(med, 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
        "mad": round(statistics.median([abs(v - med) for v in vals]), 3),
        "n": len(vals),
    }


def main():
    out_path = Path(sys.argv[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": {}, "summary": {}}
    for name, prompt in PROMPTS.items():
        warmup = one_run(name, prompt)
        runs = [one_run(name, prompt) for _ in range(RUNS)]
        doc["runs"][name] = {"warmup_discarded": warmup, "retained": runs}
        doc["summary"][name] = {
            "prompt_tps": summarise(runs, lambda r: r["timings"].get("prompt_per_second")),
            "gen_tps": summarise(runs, lambda r: r["timings"].get("predicted_per_second")),
            "ttft_ms": summarise(runs, lambda r: r["timings"].get("prompt_ms")),
            "wall_seconds": summarise(runs, lambda r: r["wall_seconds"]),
            "completion_tokens": sorted({r["usage"].get("completion_tokens") for r in runs}),
            "output_hashes": sorted({r["output_sha256"] for r in runs}),
            "finish_reasons": sorted({r["finish_reason"] for r in runs}),
        }
        print(name, json.dumps(doc["summary"][name].get("gen_tps")), flush=True)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    print("written:", out_path)


if __name__ == "__main__":
    main()
