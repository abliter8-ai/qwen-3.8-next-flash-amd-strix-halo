#!/usr/bin/env python3
"""Synchronize Prism's pinned host inputs before an integrated GPU reuses them."""
from pathlib import Path
import shutil
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch-strix-input-sync.py PRISM_SOURCE")
    path = Path(sys.argv[1]) / "src/llama-context.cpp"
    source = path.read_text()
    old = """        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);"""
    new = """        // Strix HIP reads pinned host inputs zero-copy.
        // Finish the previous ubatch before overwriting those inputs.
        ggml_backend_sched_synchronize(sched.get());

""" + old
    if source.count(new) == 1:
        print("Strix input synchronization is already applied")
        return
    if source.count(old) != 1:
        raise SystemExit("Unexpected Prism source; no changes made")
    backup = path.with_suffix(".cpp.before-strix-input-sync")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(source.replace(old, new))
    print("Applied Strix input synchronization")


if __name__ == "__main__":
    main()
