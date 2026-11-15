#!/usr/bin/env python3
"""Reproduce, from published git objects only, every hash the live server reported (see served_source_record.py).

Needs: a clone that contains the served commit, and the record directory. No GPU, no WSL, no network.

  python scripts/verify_served_revision.py [reports/served_5002/record.json]

Checks
  1. source_tree_sha256: the served commit's source files (+ the published copies of files that were untracked at
     startup) hash to the reported value with the server's own algorithm.
  2. Every module the server imports is a tracked file of that commit (untracked files in the hash are not served).
  3. Every reported artifact SHA256 equals the commit's blob, either as stored (LF) or with CRLF line endings
     (what a Windows checkout writes to disk).
Exit code 0 only if all pass.
"""

import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIRS, SOURCE_FILES = ("src", "scripts", "third_party"), ("main.py", "run_v2_socket.py")


def git(*a) -> bytes:
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, check=True).stdout


def lf(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def main():
    rec_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "reports", "served_5002", "record.json")
    rec = json.load(open(rec_path))
    rep = rec["server_reported_revision"]
    commit = rep["code"]["commit"]
    ok = True

    names = git("ls-tree", "-r", "--name-only", commit).decode().splitlines()
    tracked = sorted(n for n in names if n in SOURCE_FILES or (n.split("/")[0] in SOURCE_DIRS and
                                                                n.endswith((".py", ".sh", ".ps1")) and "__pycache__" not in n))
    files = {rel: git("show", f"{commit}:{rel}") for rel in tracked}
    for rel, info in rec["source_tree"]["untracked_at_start"].items():
        files[rel] = open(os.path.join(ROOT, info["copy"]), "rb").read()
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode() + b"\0" + lf(files[rel]) + b"\0")
    got = h.hexdigest()
    match = got == rep["code"]["source_tree_sha256"]
    ok &= match
    print(f"[{'PASS' if match else 'FAIL'}] source_tree_sha256 {got}: commit {commit[:10]} ({len(tracked)} tracked files)"
          f" + {len(rec['source_tree']['untracked_at_start'])} untracked-at-start file(s)")

    served = rec["served_modules"]["files"]
    missing = [f for f in served if f not in tracked]
    untracked_served = [f for f, i in rec["source_tree"]["untracked_at_start"].items() if i["imported_by_server"]]
    good = not missing and not untracked_served
    ok &= good
    print(f"[{'PASS' if good else 'FAIL'}] all {len(served)} served modules are tracked files of {commit[:10]}"
          + (f"; missing {missing}" if missing else "") + (f"; untracked served {untracked_served}" if untracked_served else ""))

    for rel, want in rep["artifacts_sha256"].items():
        blob = git("show", f"{commit}:{rel}")
        forms = {"LF (as stored)": hashlib.sha256(blob).hexdigest(),
                 "CRLF (Windows checkout)": hashlib.sha256(lf(blob).replace(b"\n", b"\r\n")).hexdigest()}
        hit = [k for k, v in forms.items() if v == want]
        ok &= bool(hit)
        print(f"[{'PASS' if hit else 'FAIL'}] {rel}: reported {want[:12]}... = commit blob, {hit[0] if hit else 'no form'}")
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
