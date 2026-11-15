"""Fingerprints that bind caches, fold checkpoints and exports to the exact inputs that produced them.

A cached feature matrix or a tuned fold must never silently survive a change to the preprocessing,
the transcripts, the labels/identity groups, the feature code, the training code or the base weights.
Every artifact records a fingerprint; readers recompute it and refuse a mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Dict, Iterable, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def code_hash(rel_paths: Iterable[str]) -> str:
    """Hash of source files with line endings normalised (CRLF checkouts hash like LF ones)."""
    h = hashlib.sha256()
    for rel in sorted(rel_paths):
        data = open(os.path.join(ROOT, rel), "rb").read().replace(b"\r\n", b"\n")
        h.update(rel.encode() + b"\0" + data + b"\0")
    return h.hexdigest()


def _digest(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def clip_cache_fingerprint(store) -> Dict[str, str]:
    """Identity of the preprocessed real-clip cache: ids, labels, identity groups, transcripts, per-clip
    preprocessing meta (frame times, face boxes, detect rate), audio samples and frame-array sizes, plus the
    preprocessing code. Raw frame arrays (~7 GB) are bound by size + meta rather than re-hashed every run."""
    h = hashlib.sha256()
    for cid in store.ids:
        r = store.by_id[cid]
        meta = open(os.path.join(store.cache, "meta", cid + ".json"), "rb").read()
        fr = os.path.getsize(os.path.join(store.cache, "frames", cid + ".npy"))
        au = sha256_file(os.path.join(store.cache, "audio", cid + ".npy"))
        h.update(json.dumps([cid, int(r.label), int(r.group), sha256_bytes(r.transcript.encode()),
                             sha256_bytes(meta), fr, au]).encode())
    return {"clips": h.hexdigest(), "groups_file": sha256_file(os.path.join(ROOT, "data", "rlt_identity_groups.json")),
            "preprocess_code": code_hash(["scripts/preprocess_rlt.py", "src/rlt_data.py"])}


def base_weights_fingerprint(backbone: str) -> Optional[str]:
    """Hash of the recorded base-model sources for one backbone (models/MODEL_SOURCES.json entry)."""
    p = os.path.join(ROOT, "models", "MODEL_SOURCES.json")
    if not os.path.exists(p):
        return None
    src = json.load(open(p))
    keys = {"vjepa2": ["vjepa2-vitl-fpc64-256"], "vjepa2_1": ["vjepa2_1-vitb-384"],
            "vjepa2_1_l": ["vjepa2_1-vitl-384"], "dasheng": ["dasheng-base"],
            "emotion2vec": ["emotion2vec_plus_large", "emotion2vec_plus_large-config"],
            "qwen3": ["qwen3-embedding-0.6b"]}.get(backbone)
    if not keys or any(k not in src for k in keys):
        return None
    return _digest({k: src[k] for k in keys})


TUNING_CODE = ["src/tuning.py", "src/backbones.py", "src/clip_store.py"]


def tuning_fingerprint(store, backbone: str, ladder, seed: int, k: int) -> Dict[str, str]:
    """Everything a tuned fold depends on. Stored in fold<k>.pt; resume refuses a mismatch."""
    from dataclasses import asdict

    fp = {"cache": _digest(clip_cache_fingerprint(store)), "code": code_hash(TUNING_CODE),
          "base_weights": base_weights_fingerprint(backbone) or "unknown",
          "ladder": _digest([asdict(c) for c in ladder]), "seed": str(seed), "k": str(k)}
    fp["all"] = _digest(fp)
    return fp


SOURCE_DIRS = ("src", "scripts", "third_party")
SOURCE_FILES = ("main.py", "run_v2_socket.py")


LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def in_source_scope(rel: str) -> bool:
    return rel in SOURCE_FILES or (rel.split("/")[0] in SOURCE_DIRS and rel.endswith((".py", ".sh", ".ps1"))
                                   and "__pycache__" not in rel)


def _walk_source_files():
    out = list(SOURCE_FILES)
    for d in SOURCE_DIRS:
        for base, _, files in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in base:
                continue
            out += [os.path.relpath(os.path.join(base, f), ROOT).replace(os.sep, "/") for f in files]
    return sorted(f for f in out if in_source_scope(f) and os.path.exists(os.path.join(ROOT, f)))


def _git(*args) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True, timeout=30).stdout


def tracked_blobs(commit: str = "HEAD") -> Dict[str, str]:
    """{path: git blob id} for every tracked file at `commit` (plumbing only: same answer in WSL and Windows)."""
    out = {}
    for entry in _git("ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if entry:
            meta, path = entry.split(b"\t", 1)
            out[path.decode()] = meta.split()[2].decode()
    return out


def sha256_lf_file(rel: str) -> str:
    """SHA256 of a text file with CRLF -> LF: the same on a Windows checkout, in WSL and in an LF clone."""
    return sha256_bytes(open(os.path.join(ROOT, rel), "rb").read().replace(b"\r\n", b"\n"))


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def matches_blob(rel: str, blob_id: str) -> bool:
    """Is the file on disk the committed content? Accepts the stored bytes, a CRLF checkout of them and a smudged
    git-lfs object, so the answer does not depend on which git (Windows or WSL) wrote or reads the checkout."""
    data = open(os.path.join(ROOT, rel), "rb").read()
    if git_blob_id(data) == blob_id or git_blob_id(data.replace(b"\r\n", b"\n")) == blob_id:
        return True
    blob = _git("cat-file", "blob", blob_id)
    if blob.startswith(LFS_POINTER):
        fields = dict(line.split(b" ", 1) for line in blob.splitlines() if b" " in line)
        return fields.get(b"oid", b"").decode() == "sha256:" + sha256_bytes(data) and \
            int(fields.get(b"size", b"-1")) == len(data)
    return False


def source_tree_files(tracked: Optional[Dict[str, str]] = None):
    """Source files in the tree hash: the TRACKED main.py, run_v2_socket.py and *.py/*.sh/*.ps1 under src/, scripts/,
    third_party/ (their current disk content). Untracked files are reported separately, never mixed into the hash,
    so the hash can be reproduced from a commit. Without git, every such file on disk is used."""
    if tracked is None:
        try:
            tracked = tracked_blobs()
        except Exception:
            return _walk_source_files()
    return sorted(r for r in tracked if in_source_scope(r) and os.path.exists(os.path.join(ROOT, r)))


def source_tree_fingerprint(tracked: Optional[Dict[str, str]] = None) -> str:
    """Content hash of every tracked source file that can affect training, export or serving (line endings
    normalised). Uncommitted edits to tracked files change it; untracked files do not (see git_revision)."""
    return code_hash(source_tree_files(tracked))


def git_revision() -> Dict[str, object]:
    """Commit + content-based dirty flag + tree hash, identical whichever git (Windows or WSL) made the checkout.

    dirty                    a tracked file's content differs from HEAD (CRLF and git-lfs smudging are not changes)
    modified_tracked_files   which ones (first 50)
    untracked_source_files   source-scope files on disk that git does not track, with CRLF->LF SHA256; they are not
                             in source_tree_sha256 and are not served unless imported
    """
    try:
        rev = _git("rev-parse", "HEAD").decode().strip()
        tracked = tracked_blobs(rev)
    except Exception:
        return {"commit": None, "dirty": None, "source_tree_sha256": code_hash(_walk_source_files()),
                "source_tree_scope": "all source files on disk (git unavailable)"}
    modified = [r for r, b in sorted(tracked.items())
                if not os.path.exists(os.path.join(ROOT, r)) or not matches_blob(r, b)]
    untracked = {r: sha256_lf_file(r) for r in _walk_source_files() if r not in tracked}
    return {"commit": rev, "dirty": bool(modified), "modified_tracked_files": modified[:50],
            "source_tree_sha256": source_tree_fingerprint(tracked),
            "source_tree_scope": "tracked main.py, run_v2_socket.py and *.py/*.sh/*.ps1 under src/, scripts/, "
                                 "third_party/; CRLF->LF; src.provenance.code_hash",
            "untracked_source_files": untracked}


def state_hash(state) -> str:
    """Order-independent hash of a nested {name: tensor} trainable state."""
    import torch

    h = hashlib.sha256()

    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k in sorted(obj):
                walk(f"{prefix}/{k}", obj[k])
        elif isinstance(obj, torch.Tensor):
            t = obj.detach().to("cpu").contiguous()
            h.update(prefix.encode() + str(t.dtype).encode() + str(tuple(t.shape)).encode())
            h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    walk("", state)
    return h.hexdigest()
