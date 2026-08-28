#!/usr/bin/env python3
"""IP-276 Task 10 depth ladder: fixed prompts at ~8K / ~111,411 / ~200K tokens
against the 262,144-token allocation. Uses /tokenize to hit exact token targets,
captures server timings, wall time, and output coherence.

Usage: ladder.py OUTPUT_DIR [rungs...]   (default rungs: 8000 111411 200000)
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://localhost:8080")
MODEL = os.environ.get("LLAMA_MODEL", "qwen3.8-flash-next")

SEED_PARA = (
    "Fleet log entry %d. The container ship altered course at dawn, logging "
    "position, fuel state, and reefer temperatures before the watch change. "
    "Cargo manifest checks continued through the morning as the convoy passed "
    "the headland, and the engineering team recorded vibration readings from "
    "the main shaft bearing within normal tolerance. "
)


def post(path, payload, timeout=7200):
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), time.monotonic() - t0


def count_tokens(text):
    resp, _ = post("/tokenize", {"content": text})
    return len(resp["tokens"])


def build_prompt(target_tokens):
    # grow deterministically to the target, then trim by paragraph
    n, text = 0, []
    i = 0
    per_para = None
    while True:
        i += 1
        text.append(SEED_PARA % i)
        if per_para is None and i == 50:
            per_para = count_tokens("".join(text)) / 50.0
            need = int((target_tokens * 0.98) / per_para)
            while i < need:
                i += 1
                text.append(SEED_PARA % i)
        if i >= 50 and per_para is not None:
            break
    prompt = "".join(text)
    tokens = count_tokens(prompt)
    question = ("\n\nHow many fleet log entries appear above? Answer with a "
                "number, then name one recurring measurement they record.")
    return prompt + question, tokens


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    rungs = [int(x) for x in sys.argv[2:]] or [8000, 111411, 200000]
    results = []
    for target in rungs:
        prompt, tokens = build_prompt(target)
        print(f"rung {target}: built prompt of {tokens} tokens", flush=True)
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 256,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp, wall = post("/v1/chat/completions", payload)
        t = resp.get("timings", {})
        content = resp["choices"][0]["message"].get("content") or ""
        rec = {
            "target": target,
            "prompt_tokens_built": tokens,
            "usage": resp.get("usage"),
            "prompt_tps": t.get("prompt_per_second"),
            "gen_tps": t.get("predicted_per_second"),
            "ttft_ms": t.get("prompt_ms"),
            "wall_seconds": round(wall, 1),
            "finish_reason": resp["choices"][0].get("finish_reason"),
            "output": content,
            "coherent_guess": ("entries" in content.lower() or any(c.isdigit() for c in content)),
        }
        results.append(rec)
        (out_dir / f"rung-{target}.json").write_text(json.dumps(rec, indent=2))
        print(f"rung {target}: pp={rec['prompt_tps']} tg={rec['gen_tps']} wall={rec['wall_seconds']}s finish={rec['finish_reason']}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(
        [{k: r[k] for k in ("target", "prompt_tokens_built", "prompt_tps", "gen_tps", "ttft_ms", "wall_seconds", "finish_reason", "coherent_guess")} for r in results],
        indent=2))
    print("ladder complete")


if __name__ == "__main__":
    main()
