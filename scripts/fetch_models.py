#!/usr/bin/env python3
"""Download and verify every backbone used by DeceptionX.

Only official sources are used:
  * Hugging Face repos owned by the model authors, pinned to a commit. Each
    LFS file is checked against the SHA-256 the Hub publishes for it.
  * Meta's CDN (dl.fbaipublicfiles.com) for V-JEPA 2.1, the URL listed in
    github.com/facebookresearch/vjepa2.

Large checkpoints that bundle optimizer state, EMA copies or predictors are
not downloaded whole. ``RangeTorchCheckpoint`` reads the zip directory of a
torch ``.pt`` file over HTTP range requests, parses ``data.pkl`` with an
unpickler that cannot execute code, and fetches only the tensor storages of the
sub-dict we need. Every fetched zip entry is CRC-checked by ``zipfile``.

Run inside the CUDA env:  python scripts/fetch_models.py [--only NAME ...]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import pickle
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
MAX_DOWNLOAD_BYTES_DEFAULT = 1_000_000_000  # user guardrail: nothing over 1 GB
UA = {"User-Agent": "deceptionx-fetch/1.0"}


# --------------------------------------------------------------------------- HF
@dataclass
class HFModel:
    name: str
    repo: str
    revision: str
    files: List[str]
    # Files the user explicitly approved above the 1 GB limit.
    approved_over_limit: List[str] = field(default_factory=list)


HF_MODELS: Dict[str, HFModel] = {
    "vjepa2-vitl-fpc64-256": HFModel(
        name="vjepa2-vitl-fpc64-256",
        repo="facebook/vjepa2-vitl-fpc64-256",
        revision="b3c1679b7c34d3255ef3547f27c7b226aefab26f",
        files=["config.json", "video_preprocessor_config.json", "model.safetensors"],
        approved_over_limit=["model.safetensors"],  # 1.30 GB, approved by user
    ),
    "qwen3-embedding-0.6b": HFModel(
        name="qwen3-embedding-0.6b",
        repo="Qwen/Qwen3-Embedding-0.6B",
        revision="main",  # resolved to a commit sha at run time and recorded
        files=[
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors",
        ],
        approved_over_limit=["model.safetensors"],  # 1.19 GB, approved by user
    ),
    "emotion2vec_plus_large-config": HFModel(
        name="emotion2vec_plus_large",
        repo="emotion2vec/emotion2vec_plus_large",
        revision="main",
        files=["config.yaml", "tokens.txt"],
    ),
}

RANGE_MODELS = {
    # V-JEPA 2.1 ViT-B/16 @384. Official loader uses checkpoint_key="ema_encoder".
    "vjepa2_1-vitb-384": dict(
        url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt",
        key="ema_encoder",
        strip_prefixes=("module.", "backbone."),
        out="encoder.safetensors",
    ),
    # V-JEPA 2.1 ViT-L/16 @384 - Meta's `vjepa2_1_vit_large_384` (README model table, upstream commit
    # 204698b45b3712590f06245fbfba32d3be539812: vit_large = 1024 wide, 24 blocks, 16 heads; ema_encoder).
    # The hub builder's VJEPA_BASE_URL points at localhost, so the README's official CDN link is used.
    # ~1.2 GB fp32 encoder subtree: over the 1 GB default, approved by the user on 2026-11-11.
    "vjepa2_1-vitl-384": dict(
        url="https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
        key="ema_encoder",
        strip_prefixes=("module.", "backbone."),
        out="encoder.safetensors",
        approved_limit_bytes=1_400_000_000,
    ),
    # emotion2vec+ large: the "model" state dict only (no optimizer state).
    "emotion2vec_plus_large": dict(
        url_tpl="https://huggingface.co/emotion2vec/emotion2vec_plus_large/resolve/{rev}/model.pt",
        repo="emotion2vec/emotion2vec_plus_large",
        key="model",
        strip_prefixes=(),
        out="model.safetensors",
    ),
}


def _get_json(url: str, retries: int = 6):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            wait = int(e.headers.get("Retry-After", 0) or 0) or 15 * (attempt + 1)
            print(f"    rate limited by {url.split('/')[2]}, waiting {wait}s", flush=True)
            time.sleep(wait)


# MediaPipe BlazeFace short-range face detector (the PDF's Node-1 face cropper).
# Verified against the MD5 / CRC32C that Google Cloud Storage publishes.
BLAZEFACE = dict(
    url="https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    out=os.path.join("mediapipe", "blaze_face_short_range.tflite"),
)


def fetch_blazeface() -> dict:
    import base64

    dest = os.path.join(MODELS, BLAZEFACE["out"])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(BLAZEFACE["url"], method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        hashes = dict(h.split("=", 1) for h in r.headers.get_all("x-goog-hash"))
        size = int(r.headers["Content-Length"])
    if not (os.path.exists(dest) and os.path.getsize(dest) == size):
        download(BLAZEFACE["url"], dest, size)
    md5 = base64.b64encode(hashlib.md5(open(dest, "rb").read()).digest()).decode()
    if md5 != hashes["md5"]:
        raise RuntimeError(f"BlazeFace MD5 mismatch {md5} != {hashes['md5']}")
    print(f"  [mediapipe] blaze_face_short_range.tflite ok: md5 {md5} matches GCS")
    return {"url": BLAZEFACE["url"], "size": size, "md5_b64": md5}


def resolve_revision(repo: str, revision: str) -> str:
    if len(revision) == 40:
        return revision
    return _get_json(f"https://huggingface.co/api/models/{repo}/revision/{revision}")["sha"]


def hub_tree(repo: str, sha: str) -> Dict[str, dict]:
    return {f["path"]: f for f in _get_json(f"https://huggingface.co/api/models/{repo}/tree/{sha}")}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: str) -> str:
    h = hashlib.sha1()
    size = os.path.getsize(path)
    h.update(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_against_tree(path: str, entry: dict) -> str:
    """Return a description of the check; raise on mismatch."""
    if "lfs" in entry:
        want = entry["lfs"]["oid"]
        got = sha256_file(path)
        if got != want:
            raise RuntimeError(f"SHA-256 mismatch for {path}: {got} != {want}")
        return f"sha256 {got[:16]}... matches Hub LFS oid"
    want = entry["oid"]  # git blob sha1 for non-LFS files
    got = git_blob_sha1(path)
    if got != want:
        raise RuntimeError(f"git blob sha1 mismatch for {path}: {got} != {want}")
    return f"git-blob sha1 {got[:12]} matches Hub"


def download(url: str, dest: str, expected_size: Optional[int] = None) -> None:
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total, t0, last = 0, time.time(), 0.0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if time.time() - last > 10:
                last = time.time()
                print(f"    {os.path.basename(dest)}: {total/1e6:.0f} MB ({total/1e6/(last-t0+1e-9):.0f} MB/s)", flush=True)
    if expected_size is not None and total != expected_size:
        os.remove(tmp)
        raise RuntimeError(f"size mismatch for {url}: {total} != {expected_size}")
    os.replace(tmp, dest)


def fetch_hf(m: HFModel, limit: int) -> dict:
    sha = resolve_revision(m.repo, m.revision)
    tree = hub_tree(m.repo, sha)
    out_dir = os.path.join(MODELS, m.name)
    os.makedirs(out_dir, exist_ok=True)
    record = {"repo": m.repo, "revision": sha, "files": {}}
    for fn in m.files:
        entry = tree[fn]
        size = entry.get("size", 0)
        if size > limit and fn not in m.approved_over_limit:
            raise RuntimeError(f"{m.repo}/{fn} is {size/1e9:.2f} GB > limit and not approved")
        dest = os.path.join(out_dir, fn)
        if os.path.exists(dest) and os.path.getsize(dest) == size:
            print(f"  [{m.name}] {fn} present, verifying")
        else:
            print(f"  [{m.name}] downloading {fn} ({size/1e6:.1f} MB) @ {sha[:10]}")
            download(f"https://huggingface.co/{m.repo}/resolve/{sha}/{fn}", dest, size)
        record["files"][fn] = {"size": size, "check": verify_against_tree(dest, entry)}
        print(f"    ok: {record['files'][fn]['check']}")
    return record


# --------------------------------------------------------- range-fetch torch zip
class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url: str, retries: int = 4):
        self.url, self.pos, self.fetched, self.retries = url, 0, 0, retries
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.final_url = r.geturl()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        n = len(b)
        if n == 0 or self.pos >= self.size:
            return 0
        end = min(self.pos + n, self.size) - 1
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.final_url, headers={**UA, "Range": f"bytes={self.pos}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise RuntimeError(f"server ignored Range header (status {r.status})")
                    data = r.read()
                break
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        b[: len(data)] = data
        self.pos += len(data)
        self.fetched += len(data)
        return len(data)


_STORAGE_DTYPES = {
    "FloatStorage": "float32",
    "HalfStorage": "float16",
    "BFloat16Storage": "bfloat16",
    "DoubleStorage": "float64",
    "LongStorage": "int64",
    "IntStorage": "int32",
    "ShortStorage": "int16",
    "CharStorage": "int8",
    "ByteStorage": "uint8",
    "BoolStorage": "bool",
}


class _Opaque:
    """Inert stand-in for any non-tensor class referenced by the pickle."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


@dataclass
class _StorageRef:
    key: str
    dtype: str
    numel: int


@dataclass
class _TensorRef:
    storage: _StorageRef
    offset: int
    size: tuple
    stride: tuple


class SafeTorchUnpickler(pickle.Unpickler):
    """Parses torch's data.pkl without importing or calling arbitrary objects."""

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch" and name in _STORAGE_DTYPES:
            return ("dtype", _STORAGE_DTYPES[name])
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda st, off, size, stride, *a: _TensorRef(st, off, tuple(size), tuple(stride))
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *a: data
        return _Opaque

    def persistent_load(self, pid):
        typename, storage_type, key, _location, numel = pid
        if typename != "storage":
            raise pickle.UnpicklingError(f"unexpected persistent id {typename}")
        dtype = storage_type[1] if isinstance(storage_type, tuple) else None
        if dtype is None:
            raise pickle.UnpicklingError(f"unsupported storage type {storage_type}")
        return _StorageRef(str(key), dtype, int(numel))


class RangeTorchCheckpoint:
    def __init__(self, url: str):
        self.raw = HTTPRangeFile(url)
        self.zf = zipfile.ZipFile(io.BufferedReader(self.raw, buffer_size=1 << 16))
        names = self.zf.namelist()
        pkl = next(n for n in names if n.endswith("/data.pkl"))
        self.prefix = pkl[: -len("data.pkl")]
        self.obj = SafeTorchUnpickler(io.BytesIO(self.zf.read(pkl))).load()
        bo = self.prefix + "byteorder"
        if bo in names and self.zf.read(bo).strip() != b"little":
            raise RuntimeError("big-endian checkpoint not supported")

    def subtree_bytes(self, key: str) -> int:
        refs = {t.storage.key for t in self.obj[key].values() if isinstance(t, _TensorRef)}
        return sum(self.zf.getinfo(f"{self.prefix}data/{k}").file_size for k in refs)

    def load_subtree(self, key: str):
        import torch

        tensors = {}
        cache: Dict[str, "torch.Tensor"] = {}
        for name, ref in self.obj[key].items():
            if not isinstance(ref, _TensorRef):
                continue
            sk = ref.storage.key
            if sk not in cache:
                data = self.zf.read(f"{self.prefix}data/{sk}")  # zipfile checks CRC-32
                dt = getattr(torch, ref.storage.dtype)
                cache[sk] = torch.frombuffer(bytearray(data), dtype=dt)
                assert cache[sk].numel() == ref.storage.numel, f"numel mismatch for storage {sk}"
            t = torch.as_strided(cache[sk], ref.size, ref.stride, ref.offset).clone()
            tensors[name] = t
        return tensors


def fetch_range(name: str, spec: dict, limit: int) -> dict:
    from safetensors.torch import save_file

    if "url_tpl" in spec:
        rev = resolve_revision(spec["repo"], "main")
        url = spec["url_tpl"].format(rev=rev)
    else:
        rev, url = None, spec["url"]
    out_dir = os.path.join(MODELS, name)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, spec["out"])
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(out) and os.path.exists(manifest_path):
        man = json.load(open(manifest_path))
        if sha256_file(out) == man.get("output_sha256"):
            print(f"  [{name}] present and matches manifest")
            return man
    print(f"  [{name}] reading checkpoint index from {url}")
    ck = RangeTorchCheckpoint(url)
    need = ck.subtree_bytes(spec["key"])
    print(f"    full file {ck.raw.size/1e9:.2f} GB; '{spec['key']}' subtree {need/1e6:.1f} MB")
    limit = max(limit, spec.get("approved_limit_bytes", 0))
    if need > limit:
        raise RuntimeError(f"{name}: needed bytes {need} exceed limit {limit}")
    t0 = time.time()
    sd = ck.load_subtree(spec["key"])
    clean = {}
    for k, v in sd.items():
        for p in spec["strip_prefixes"]:
            k = k.replace(p, "")
        clean[k] = v.contiguous()
    save_file(clean, out)
    man = {
        "source_url": url,
        "hub_revision": rev,
        "checkpoint_key": spec["key"],
        "source_size_bytes": ck.raw.size,
        "bytes_fetched": ck.raw.fetched,
        "num_tensors": len(clean),
        "num_params": int(sum(v.numel() for v in clean.values())),
        "verification": "zip CRC-32 checked for every fetched entry; storage numel checked",
        "output_sha256": sha256_file(out),
        "fetch_seconds": round(time.time() - t0, 1),
    }
    json.dump(man, open(manifest_path, "w"), indent=2)
    print(f"    ok: {man['num_tensors']} tensors, {man['num_params']/1e6:.1f}M params, fetched {man['bytes_fetched']/1e6:.1f} MB")
    return man


def verify_dasheng() -> dict:
    repo = "mispeech/dasheng-base"
    tree = hub_tree(repo, resolve_revision(repo, "main"))
    d = os.path.join(MODELS, "dasheng-base")
    res = {}
    for fn in ["config.json", "configuration_dasheng.py", "feature_extraction_dasheng.py",
               "modeling_dasheng.py", "preprocessor_config.json", "model.safetensors"]:
        res[fn] = verify_against_tree(os.path.join(d, fn), tree[fn])
        print(f"  [dasheng-base] {fn}: {res[fn]}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--limit-bytes", type=int, default=MAX_DOWNLOAD_BYTES_DEFAULT)
    args = ap.parse_args()
    want = lambda n: args.only is None or n in args.only  # noqa: E731
    src_path = os.path.join(MODELS, "MODEL_SOURCES.json")
    # Merge: a partial run (--only) must not drop the records of models fetched earlier.
    report = json.load(open(src_path)) if os.path.exists(src_path) else {}
    if want("mediapipe"):
        report["mediapipe-blazeface"] = fetch_blazeface()
    if want("dasheng-base"):
        report["dasheng-base"] = verify_dasheng()
    for key, m in HF_MODELS.items():
        if want(key) or want(m.name):
            report[key] = fetch_hf(m, args.limit_bytes)
    for name, spec in RANGE_MODELS.items():
        if want(name):
            report[name] = fetch_range(name, spec, args.limit_bytes)
    with open(src_path, "w") as f:
        json.dump(report, f, indent=2)
    print("all requested models verified")


if __name__ == "__main__":
    sys.exit(main())
