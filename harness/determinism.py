#!/usr/bin/env python3
"""IP-276 deterministic comparison set.

Runs 20 fixed prompts at temperature 0 against the live server and writes
{prompt_id: {content_sha256, reasoning_sha256, content}} to OUTPUT_JSON.
Compare two arms (baseline vs MTP) with:  determinism.py compare A.json B.json
"""
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://localhost:8080")
MODEL = os.environ.get("LLAMA_MODEL", "qwen3.8-flash-next")

PROMPTS = [
    "Explain the CAP theorem in three sentences.",
    "Write a haiku about NVMe drives.",
    "What is 391 * 27? Show the answer only.",
    "List the planets in order from the sun, comma-separated.",
    "Translate 'the ship sails at dawn' into German and French.",
    "Write a Python one-liner that reverses the words in a string.",
    "Name the time complexity of binary search and why, in one sentence.",
    "Summarise the plot of Moby-Dick in two sentences.",
    "What year did the Berlin Wall fall? Answer with the year only.",
    "Write a SQL query selecting the top 5 customers by total order value.",
    "Explain what a mutex is to a ten-year-old, two sentences.",
    "Give the chemical formula for table salt and for water.",
    "Write a limerick about a GPU that ran out of memory.",
    "Convert 100 degrees Fahrenheit to Celsius, show one decimal.",
    "Name three Hebridean islands.",
    "Write a regex that matches an IPv4 address.",
    "What does HTTP status 418 mean? One sentence.",
    "Alphabetise: kiwi, banana, apple, cherry, mango.",
    "Write a two-line bash script that counts files in /tmp.",
    "State Ohm's law as a formula and in words.",
]


def run_prompt(text, thinking):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": 512,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.loads(r.read().decode())
    msg = resp["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return {
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
        "content": content,
        "finish_reason": resp["choices"][0].get("finish_reason"),
    }


def capture(out_path):
    doc = {}
    for i, p in enumerate(PROMPTS):
        # half with thinking off, half on — both modes must be deterministic
        doc[f"p{i:02d}"] = run_prompt(p, thinking=(i % 2 == 1))
        print(f"p{i:02d} done", flush=True)
    Path(out_path).write_text(json.dumps(doc, indent=2) + "\n")
    print("written:", out_path)


def compare(a_path, b_path):
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    mismatches = []
    for k in sorted(a):
        if a[k]["content_sha256"] != b[k].get("content_sha256"):
            mismatches.append(k)
    print(json.dumps({
        "total": len(a),
        "equal": len(a) - len(mismatches),
        "mismatched": mismatches,
    }, indent=2))
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        capture(sys.argv[1])
