#!/usr/bin/env python3
"""Record EXACTLY which source and artifacts the process listening on a port is serving (run inside WSL, next to it).

The live server reports `server_revision` = {commit, dirty, source_tree_sha256} + artifact SHA256s, computed once
at startup from the working tree on disk. That alone does not let anyone else reproduce it, because:
  * `dirty` comes from git inside WSL, which lists a Windows (CRLF, git-lfs) checkout as modified;
  * `source_tree_sha256` covers every *.py/*.sh/*.ps1 under src/, scripts/, third_party/, including untracked
    files that were on disk at startup;
  * artifact hashes are of the raw disk bytes, which are CRLF for text files in a Windows checkout.
This script asks the running server for its revision, then writes a self-contained record:
  reports/served_<port>/record.json             process, reported revision verbatim, per-file hashes, served modules,
                                                artifact hashes in every form, and the outcome of each check
  reports/served_<port>/untracked_at_start/...  exact copies of files that were untracked at startup but are in the hash
`scripts/verify_served_revision.py` then reproduces every reported hash from published git objects + those copies.

  bash scripts/run.sh scripts/served_source_record.py [--port 5002] [--untracked-at-start REL=PATH ...]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.provenance import SOURCE_DIRS, SOURCE_FILES  # noqa: E402

ARTIFACTS = ("models/ensemble_deploy.json", "models/branch2_xgboost.json", "models/branch2_digest_schema.json",
             "models/vjepa2_1_l_lora/deploy_meta.json", "models/vjepa2_1-vitl-384/manifest.json",
             "models/pilot_webcam/dasheng/deploy_meta.json", "models/pilot_webcam/vjepa2_1_l/deploy_meta.json",
             "models/pilot_webcam/emotion2vec/deploy_meta.json", "models/pilot_webcam/fusion.json")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def lf(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def git(*a) -> bytes:
    return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, check=True).stdout


def tree_files_at(commit: str):
    names = git("ls-tree", "-r", "--name-only", commit).decode().splitlines()
    return sorted(n for n in names if n in SOURCE_FILES or (n.split("/")[0] in SOURCE_DIRS and
                                                             n.endswith((".py", ".sh", ".ps1")) and "__pycache__" not in n))


def tree_hash(files: dict) -> str:
    """Identical to src.provenance.code_hash over source_tree_files()."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode() + b"\0" + lf(files[rel]) + b"\0")
    return h.hexdigest()


def listener_pid(port: int):
    out = subprocess.run(["ss", "-ltnpH", f"sport = :{port}"], capture_output=True, text=True).stdout
    return int(out.split("pid=")[1].split(",")[0]) if "pid=" in out else None


def process_info(pid: int):
    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    boot = float(open("/proc/stat").read().split("btime ")[1].split()[0])
    start = boot + int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[19]) / hz
    return {"pid": pid, "cmdline": open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode().strip(),
            "cwd": os.readlink(f"/proc/{pid}/cwd"), "exe": os.readlink(f"/proc/{pid}/exe"),
            "started_unix": start, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start))}


def ask_revision(port: int):
    """A real jepa_video request on the generated golden fixture (no person); replies carry server_revision."""
    from scripts.make_golden_fixture import load_payload

    p, _ = load_payload()
    req = {"action": "jepa_video", "frames": p["frames"], "timestamps": p["timestamps"],
           "request_id": "served-source-record", "source_id": "golden-fixture"}
    s = socket.create_connection(("127.0.0.1", port), timeout=120)
    s.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = s.recv(1 << 20)
        if not chunk:
            break
        buf += chunk
    r = json.loads(buf)
    if r.get("status") != "success" or "server_revision" not in r:
        raise SystemExit(f"server did not return a revision: {r.get('error')}")
    return r["server_revision"]


def served_modules(entry: str = "run_v2_socket.py"):
    """Repo files reachable by imports from the server entry point, including imports inside functions."""
    seen, todo = set(), [entry]
    while todo:
        rel = todo.pop()
        if rel in seen or not os.path.exists(os.path.join(ROOT, rel)):
            continue
        seen.add(rel)
        tree = ast.parse(open(os.path.join(ROOT, rel), "rb").read())
        pkg = os.path.dirname(rel).replace("/", ".")
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = pkg.split(".")[: len(pkg.split(".")) - node.level + 1] if pkg else []
                    base = ".".join([p for p in parts if p] + ([base] if base else []))
                names = [base] + [f"{base}.{a.name}" for a in node.names]
            for n in names:
                for cand in (n.replace(".", "/") + ".py", n.replace(".", "/") + "/__init__.py"):
                    if cand.split("/")[0] in SOURCE_DIRS or cand in SOURCE_FILES:
                        todo.append(cand)
    return sorted(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5002)
    ap.add_argument("--untracked-at-start", nargs="*", default=[],
                    help="REL=PATH: exact bytes of a file that was untracked when the server started")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "reports", f"served_{args.port}")
    pid = listener_pid(args.port)
    if pid is None:
        raise SystemExit(f"nothing is listening on {args.port}")
    proc = process_info(pid)
    if os.path.realpath(proc["cwd"]) != os.path.realpath(ROOT):
        raise SystemExit(f"port {args.port} is served from {proc['cwd']}, not this checkout ({ROOT})")
    reported = ask_revision(args.port)
    commit = reported["code"]["commit"]

    tracked = tree_files_at(commit)
    files = {rel: git("show", f"{commit}:{rel}") for rel in tracked}
    per_file = {}
    for rel in tracked:
        disk = os.path.join(ROOT, rel)
        per_file[rel] = {"git_blob": git("rev-parse", f"{commit}:{rel}").decode().strip(),
                         "sha256_lf": sha(lf(files[rel])),
                         "disk_now_sha256_lf": sha(lf(open(disk, "rb").read())) if os.path.exists(disk) else None,
                         "disk_mtime_after_start": os.path.exists(disk) and os.path.getmtime(disk) > proc["started_unix"]}
    extra = {}
    for spec in args.untracked_at_start:
        rel, path = spec.split("=", 1)
        extra[rel] = open(path, "rb").read()
    candidates = {"commit_only": tree_hash(files), "commit_plus_untracked_at_start": tree_hash({**files, **extra})}
    matched = [k for k, v in candidates.items() if v == reported["code"]["source_tree_sha256"]]

    artifacts = {}
    for rel, rep in reported["artifacts_sha256"].items():
        blob = git("show", f"{commit}:{rel}")
        crlf = lf(blob).replace(b"\n", b"\r\n")
        artifacts[rel] = {"server_reported_sha256": rep, "git_blob_sha256": sha(blob), "git_blob_crlf_sha256": sha(crlf),
                          "reported_bytes_are": "git blob (LF)" if rep == sha(blob) else
                          "git blob with CRLF line endings (Windows checkout)" if rep == sha(crlf) else "NOT REPRODUCIBLE"}

    served = served_modules()
    record = {
        "what": f"Exact source and artifacts served on port {args.port}",
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "process": proc, "server_reported_revision": reported,
        "served_code_commit": commit,
        "source_tree": {"algorithm": "sha256 over sorted(path + NUL + bytes(CRLF->LF) + NUL) for main.py, run_v2_socket.py "
                                     "and *.py/*.sh/*.ps1 under src/, scripts/, third_party/ (src.provenance.code_hash)",
                        "reported": reported["code"]["source_tree_sha256"], "reconstructions": candidates,
                        "reproduced_by": matched,
                        "untracked_at_start": {rel: {"sha256_lf": sha(lf(b)),
                                                     "copy": f"reports/served_{args.port}/untracked_at_start/{rel}",
                                                     "imported_by_server": rel in served} for rel, b in extra.items()},
                        "tracked_files_at_commit": per_file},
        "served_modules": {"entry": "run_v2_socket.py", "files": served,
                           "all_identical_to_commit": all(per_file.get(f, {}).get("sha256_lf") ==
                                                          per_file.get(f, {}).get("disk_now_sha256_lf") for f in served),
                           "changed_on_disk_after_start": [f for f in served if per_file.get(f, {}).get("disk_mtime_after_start")]},
        "artifacts": artifacts,
        "dirty_flag": ({"reported": reported["code"]["dirty"],
                        "rule": "content-based (src.provenance.git_revision): CRLF checkouts and git-lfs objects are not changes",
                        "modified_tracked_files": reported["code"]["modified_tracked_files"],
                        "untracked_source_files_not_in_hash": reported["code"].get("untracked_source_files", {})}
                       if "modified_tracked_files" in reported["code"] else
                       {"reported": reported["code"]["dirty"],
                        "cause": "computed by git inside WSL (core.autocrlf unset, no git-lfs) on a checkout made by Windows "
                                 "git (core.autocrlf=true, git-lfs): every CRLF text file and the LFS-tracked dasheng-base "
                                 "weights look modified. It is not evidence of code changes; the source_tree reconstruction is."}),
    }
    os.makedirs(out_dir, exist_ok=True)
    for rel, b in extra.items():
        dst = os.path.join(out_dir, "untracked_at_start", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(b)
    json.dump(record, open(os.path.join(out_dir, "record.json"), "w"), indent=1)
    print(json.dumps({"pid": pid, "started_utc": proc["started_utc"], "commit": commit, "tree_reproduced_by": matched,
                      "served_modules": len(served), "served_modules_identical": record["served_modules"]["all_identical_to_commit"],
                      "artifacts": {k: v["reported_bytes_are"] for k, v in artifacts.items()}}, indent=1))
    if not matched:
        raise SystemExit("reported source_tree_sha256 NOT reproduced")


if __name__ == "__main__":
    main()
