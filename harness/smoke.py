#!/usr/bin/env python3
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PLAN_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://localhost:8080")
CHAT_URL = f"{BASE_URL}/v1/chat/completions"
MODEL = os.environ.get("LLAMA_MODEL", "qwen3.8-flash-next")
# any local JPEG works; set VISION_FIXTURE to point at yours
IMAGE_PATH = Path(os.environ.get("VISION_FIXTURE", PLAN_ROOT / "fixtures/vision.jpg"))
NORMAL_FINISH_REASONS = {"stop", "length", "tool_calls"}


def http_json(method, url, payload=None, timeout=600):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error
    return result, time.monotonic() - started


def message_text(response):
    message = response["choices"][0]["message"]
    parts = []
    for key in ("reasoning_content", "content"):
        value = message.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
    return "\n".join(parts).strip()


def require_normal_finish(response):
    reason = response["choices"][0].get("finish_reason")
    if reason not in NORMAL_FINISH_REASONS:
        raise AssertionError(f"abnormal finish_reason: {reason!r}")


def require_clean_text(text):
    if not text.strip():
        raise AssertionError("empty answer")
    if re.search(r"\bnan\b", text, flags=re.IGNORECASE):
        raise AssertionError("answer contains NaN")
    compact = re.sub(r"\s+", "", text)
    if compact and compact.count("/") / len(compact) > 0.5:
        raise AssertionError("answer consists mostly of slash tokens")
    if re.search(r"(?:^|\s)(\S+)(?:\s+\1){20,}(?:\s|$)", text):
        raise AssertionError("one token repeats more than 20 times")


def run_chat(name, payload, validator, records):
    record = {"name": name, "request": payload, "passed": False}
    try:
        response, elapsed = http_json("POST", CHAT_URL, payload)
        record.update({"response": response, "elapsed_seconds": elapsed})
        require_normal_finish(response)
        validator(response)
        record["passed"] = True
        record["reason"] = "passed"
    except Exception as error:  # Preserve all evidence before returning failure.
        record["reason"] = f"{type(error).__name__}: {error}"
    records.append(record)


def base_payload(messages, **extra):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
    }
    payload.update(extra)
    return payload


def validate_text(response):
    require_clean_text(message_text(response))


def validate_rowan(response):
    text = message_text(response)
    require_clean_text(text)
    if "rowan" not in text.lower():
        raise AssertionError("answer did not recall Rowan")


def validate_tool(response):
    calls = response["choices"][0]["message"].get("tool_calls") or []
    if len(calls) != 1:
        raise AssertionError(f"expected one tool call, got {len(calls)}")
    function = calls[0].get("function") or {}
    if function.get("name") != "lookup_weather":
        raise AssertionError(f"wrong tool name: {function.get('name')!r}")
    arguments = function.get("arguments", "")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if arguments.get("city") != "Dublin":
        raise AssertionError(f"wrong city arguments: {arguments!r}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke.py OUTPUT_JSON")
    output_path = Path(sys.argv[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not IMAGE_PATH.is_file():
        raise SystemExit(f"missing vision fixture: {IMAGE_PATH}")

    records = []
    run_chat(
        "text",
        base_payload([{
            "role": "user",
            "content": "Explain consistent hashing in exactly two sentences.",
        }]),
        validate_text,
        records,
    )
    run_chat(
        "multi_turn",
        base_payload([
            {"role": "user", "content": "My codename is Rowan. Acknowledge it."},
            {"role": "assistant", "content": "Acknowledged."},
            {"role": "user", "content": "What is my codename? Answer with the codename."},
        ]),
        validate_rowan,
        records,
    )
    run_chat(
        "reasoning_off",
        base_payload(
            [{"role": "user", "content": "What is 37 multiplied by 19? Give only the result."}],
            chat_template_kwargs={"enable_thinking": False},
        ),
        validate_text,
        records,
    )
    run_chat(
        "reasoning_on",
        base_payload(
            [{"role": "user", "content": "A farmer has 17 rows of 23 trees. How many trees?"}],
            chat_template_kwargs={"enable_thinking": True},
        ),
        validate_text,
        records,
    )
    tool = {
        "type": "function",
        "function": {
            "name": "lookup_weather",
            "description": "Return current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
    run_chat(
        "tool_call",
        base_payload(
            [{"role": "user", "content": "Use the tool to look up weather for Dublin."}],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "lookup_weather"}},
        ),
        validate_tool,
        records,
    )
    image_b64 = base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
    run_chat(
        "vision",
        base_payload([{
            "role": "user",
            "content": [
                {"type": "text", "text": "What animal shape appears in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }]),
        validate_text,
        records,
    )

    corruption = {"name": "corruption_guard", "passed": False}
    try:
        for record in records:
            if not record["passed"]:
                raise AssertionError(f"upstream check failed: {record['name']}")
            if record["name"] != "tool_call":
                require_clean_text(message_text(record["response"]))
        corruption.update({"passed": True, "reason": "passed"})
    except Exception as error:
        corruption["reason"] = f"{type(error).__name__}: {error}"
    records.append(corruption)

    failed = [record["name"] for record in records if not record["passed"]]
    document = {
        "base_url": BASE_URL,
        "model": MODEL,
        "created_unix": time.time(),
        "checks": records,
        "summary": {
            "total": len(records),
            "passed": len(records) - len(failed),
            "failed": len(failed),
            "failed_checks": failed,
        },
    }
    output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document["summary"], indent=2))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
