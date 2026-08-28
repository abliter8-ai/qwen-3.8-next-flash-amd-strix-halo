#!/usr/bin/env python3
"""Executable code-quality probe. Eight tasks; the model's code is extracted and
run against held tests in a subprocess. Usage: code-qual.py OUTPUT_JSON"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("LLAMA_BASE_URL", "http://localhost:8080") + "/v1/chat/completions"

TASKS = [
    ("balanced", "Write a Python function is_balanced(s) that returns True if brackets ()[]{} in s are balanced. Output ONLY the code, no explanation.",
     "assert is_balanced('([]{})'); assert not is_balanced('(]'); assert is_balanced(''); assert not is_balanced('((('); assert is_balanced('a(b)c[d]')"),
    ("rle", "Write a Python function rle(s) that run-length encodes a string: 'aaabcc' -> 'a3b1c2'. Output ONLY the code.",
     "assert rle('aaabcc')=='a3b1c2'; assert rle('')==''; assert rle('a')=='a1'; assert rle('aabbaa')=='a2b2a2'"),
    ("merge_intervals", "Write a Python function merge_intervals(iv) that merges overlapping [start,end] intervals given as a list of lists, returning them sorted. Output ONLY the code.",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]; assert merge_intervals([[1,4],[4,5]])==[[1,5]]; assert merge_intervals([])==[]"),
    ("lru", "Write a Python class LRU(capacity) with get(k) and put(k,v) methods, evicting the least-recently-used key when over capacity. get returns -1 when missing and counts as use. Output ONLY the code.",
     "c=LRU(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1; assert c.get(3)==3; assert c.get(1)==1"),
    ("csv_sum", "Write a Python function col_sum(text, col) that takes CSV text (first row is a header) and a column name, returning the float sum of that column. Output ONLY the code.",
     "t='a,b\\n1,2.5\\n3,4.5\\n'; assert col_sum(t,'a')==4.0; assert col_sum(t,'b')==7.0"),
    ("regex_ip", "Write a Python function find_ips(text) returning all valid dotted-quad IPv4 addresses (each octet 0-255) found in text, in order. Output ONLY the code.",
     "assert find_ips('at 203.0.113.7 and 999.1.1.1 and 198.51.100.254')==['203.0.113.7','198.51.100.254']"),
    ("fib_arith", "Write a Python function f(n) returning the n-th Fibonacci number (f(0)=0, f(1)=1) iteratively, then ALSO define ANSWER = f(37) * 3 - 1000. Output ONLY the code.",
     "assert f(10)==55; assert ANSWER==24157817*3-1000"),
    ("topo", "Write a Python function topo(n, edges) returning any valid topological order of nodes 0..n-1 for the DAG given by edge pairs (u,v) meaning u before v, or [] if cyclic. Output ONLY the code.",
     "o=topo(4,[(0,1),(1,2),(0,2),(2,3)]); assert o.index(0)<o.index(1)<o.index(2)<o.index(3); assert topo(2,[(0,1),(1,0)])==[]"),
]


def ask(prompt):
    payload = {"model": os.environ.get("LLAMA_MODEL", "qwen3.8-flash-next"), "messages": [{"role": "user", "content": prompt}],
               "temperature": 0, "max_tokens": 1200, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or ""


def extract(code):
    m = re.findall(r"```(?:python)?\n(.*?)```", code, re.S)
    return m[0] if m else code


def run_case(code, test):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + test + "\nprint('PASS')\n")
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=30)
        ok = r.returncode == 0 and "PASS" in r.stdout
        return ok, (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "")[:160]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        Path(path).unlink(missing_ok=True)


def main():
    out = []
    for name, prompt, test in TASKS:
        t0 = time.monotonic()
        raw = ask(prompt)
        code = extract(raw)
        ok, err = run_case(code, test)
        out.append({"task": name, "pass": ok, "err": err,
                    "secs": round(time.monotonic() - t0, 1), "chars": len(code)})
        print(name, "PASS" if ok else f"FAIL ({err})", flush=True)
    passed = sum(1 for r in out if r["pass"])
    doc = {"passed": passed, "total": len(out), "results": out}
    Path(sys.argv[1]).write_text(json.dumps(doc, indent=2))
    print(f"{passed}/{len(out)}")


if __name__ == "__main__":
    main()
