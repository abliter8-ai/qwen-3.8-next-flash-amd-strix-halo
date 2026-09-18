#!/usr/bin/env python3
"""Bounded direct checks for the Halogen + patched Prism recipe; no dependencies."""
import argparse
import base64
import json
import struct
import time
import urllib.request
import zlib
from pathlib import Path


def red_png():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = (b"\x00" + b"\xff\x00\x00" * 224) * 224
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 224, 224, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(data).decode()


def chat(url, model, content, stream=False):
    payload = {
        "model": model, "messages": [{"role": "user", "content": content}],
        "max_tokens": 128, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False}, "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=180) as response:
        if not stream:
            result = json.load(response)
            answer = result["choices"][0]["message"].get("content") or ""
            return {"answer": answer.strip(), "seconds": round(time.monotonic() - started, 3),
                    "usage": result.get("usage"), "timings": result.get("timings")}
        answer, usage, done, first = "", None, False, None
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {}).get("content") or ""
                if delta and first is None:
                    first = time.monotonic() - started
                answer += delta
        return {"answer": answer.strip(), "seconds": round(time.monotonic() - started, 3),
                "ttft_seconds": first, "done": done, "usage": usage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-url", default="http://127.0.0.1:8731")
    parser.add_argument("--bonsai-url", default="http://127.0.0.1:8080")
    parser.add_argument("--qwen-model", default="halogen-qwen3.8-flash-next")
    parser.add_argument("--bonsai-model", default="ternary-bonsai-2-27b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []

    def check(label, url, model, prompt, expected=None, stream=False):
        try:
            record = chat(url, model, prompt, stream=stream)
            answer = record["answer"].strip().rstrip(".").lower()
            passed = bool(answer) if expected is None else answer == expected.lower()
            if stream:
                passed = passed and record["done"] and bool(record["usage"])
            record.update(label=label, passed=passed)
        except Exception as error:
            record = {"label": label, "passed": False, "error": str(error)}
        records.append(record)
        print(json.dumps(record), flush=True)

    # Changing from an unrelated prompt to two longer lookups exposed stale HIP inputs.
    check("bonsai-prior-prompt", args.bonsai_url, args.bonsai_model,
          "Explain why the sky is blue in two sentences.")
    document = "\n".join(f"Record {i:04d}: access code is {100000 + 137 * i}." for i in range(60))
    for index in (10, 40):
        prompt = document + f"\nWhat is the access code for Record {index:04d}? Reply with only the code."
        check(f"bonsai-exact-lookup-{index}", args.bonsai_url, args.bonsai_model,
              prompt, str(100000 + 137 * index))
    check("bonsai-arithmetic", args.bonsai_url, args.bonsai_model,
          "What is 17 times 23? Reply with only the number.", "391")
    check("qwen-arithmetic", args.qwen_url, args.qwen_model,
          "What is 17 times 23? Reply with only the number.", "391")
    check("bonsai-vision", args.bonsai_url, args.bonsai_model, [
        {"type": "text", "text": "Name the solid color in this image. Reply with one word."},
        {"type": "image_url", "image_url": {"url": red_png()}},
    ], "red")
    check("bonsai-stream", args.bonsai_url, args.bonsai_model,
          "What is 8 times 9? Reply with only the number.", "72", stream=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checks": records}, indent=2) + "\n")
    raise SystemExit(0 if all(record["passed"] for record in records) else 1)


if __name__ == "__main__":
    main()
