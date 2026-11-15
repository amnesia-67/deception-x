#!/usr/bin/env python3
"""V-JEPA 2.1 ViT-L encoder from Hugging Face re-uploads, accepted only after cross-verification.

Why: Meta publishes V-JEPA 2.1 only at dl.fbaipublicfiles.com, whose TLS certificate expired on
2026-11-11 23:59:59 UTC. Meta has no V-JEPA 2.1 on Hugging Face. The user approved using a third-party
re-upload ONLY if it passes all of the following (safetensors only; no repository code is executed):

  1. Canary: the SAME uploader's ViT-B encoder (mnmly/vjepa2.1-vit-base-384) must be bit-identical, tensor by
     tensor, to the ViT-B ema_encoder we range-fetched from Meta under valid TLS (models/vjepa2_1-vitb-384).
  2. Integrity: every downloaded file must match the sha256 (LFS oid) Hugging Face reports at a pinned revision.
  3. Independent agreement: mnmly's ViT-L encoder must equal, tensor by tensor, the ViT-L encoder in a second,
     independently converted upload (apiantonio/vjepa2.1-vit-large-384: Transformers layout with separate
     q/k/v, mapped back to the official qkv layout).
  4. Architecture: 302 tensors, 24 blocks, width 1024, strict load into third_party/vjepa2_1 vit_large.

If any check fails nothing is written to models/. When Meta's certificate is valid again, run
`scripts/verify_vitl_official.py` to compare the accepted file with Meta's official checkpoint.

  bash scripts/run.sh scripts/fetch_vitl_mirror.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fetch_models import MODELS, download, hub_tree, resolve_revision, sha256_file  # noqa: E402

PRIMARY = ("mnmly/vjepa2.1-vit-large-384", "2594eca8e133", "model.safetensors")
CANARY = ("mnmly/vjepa2.1-vit-base-384", "ee3ee8b2a67e", "model.safetensors")
SECOND = ("apiantonio/vjepa2.1-vit-large-384", "d6cfdbdd8187", "model.safetensors")
STAGE = os.path.join(os.environ.get("DECEPTIONX_CACHE", os.path.expanduser("~/deceptionx_cache")), "mirror_verify")
OUT_DIR = os.path.join(MODELS, "vjepa2_1-vitl-384")


def fetch(repo, short_rev, fn):
    full = resolve_revision(repo, short_rev)
    assert full.startswith(short_rev), (repo, full)
    entry = hub_tree(repo, full)[fn]
    oid, size = entry["lfs"]["oid"], entry["lfs"]["size"]
    dest = os.path.join(STAGE, repo.replace("/", "__") + "__" + fn)
    if not (os.path.exists(dest) and os.path.getsize(dest) == size and sha256_file(dest) == oid):
        print(f"  downloading {repo}@{full[:12]}/{fn} ({size/1e6:.0f} MB)", flush=True)
        download(f"https://huggingface.co/{repo}/resolve/{full}/{fn}", dest, size)
    got = sha256_file(dest)
    if got != oid:
        raise SystemExit(f"FAIL integrity: {repo}/{fn} sha256 {got} != Hugging Face LFS oid {oid}")
    print(f"  ok integrity: {repo}/{fn} sha256 {got[:16]}... matches Hugging Face", flush=True)
    return dest, {"repo": repo, "revision": full, "file": fn, "size": size, "sha256": oid}


def load(path):
    from safetensors.torch import load_file

    return load_file(path)


def same(a, b, what):
    import torch

    if set(a) != set(b):
        raise SystemExit(f"FAIL {what}: key sets differ ({sorted(set(a) ^ set(b))[:5]})")
    bad = [k for k in a if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype or not torch.equal(a[k], b[k])]
    if bad:
        raise SystemExit(f"FAIL {what}: {len(bad)} tensors differ, e.g. {bad[:3]}")
    print(f"  ok {what}: {len(a)} tensors bit-identical", flush=True)


def official_from_transformers(sd):
    """apiantonio Transformers layout -> official vision_transformer layout (encoder only)."""
    import torch

    out = {}
    e = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    ren = {"embeddings.img_mod_embed": "img_mod_embed", "embeddings.video_mod_embed": "video_mod_embed",
           "embeddings.patch_embeddings.proj.weight": "patch_embed.proj.weight",
           "embeddings.patch_embeddings.proj.bias": "patch_embed.proj.bias",
           "embeddings.patch_embeddings_img.proj.weight": "patch_embed_img.proj.weight",
           "embeddings.patch_embeddings_img.proj.bias": "patch_embed_img.proj.bias"}
    for k, v in e.items():
        if k in ren:
            out[ren[k]] = v
        elif k.startswith("norms_block."):
            out[k] = v
        elif k.startswith("layer."):
            _, i, rest = k.split(".", 2)
            if rest.startswith("attention.") and rest.split(".")[1] in ("query", "key", "value"):
                continue
            rest = (rest.replace("attention.proj", "attn.proj"))
            out[f"blocks.{i}.{rest}"] = v
        else:
            raise SystemExit(f"FAIL mapping: unexpected encoder key {k}")
    n_layers = 1 + max(int(k.split(".")[1]) for k in e if k.startswith("layer."))
    for i in range(n_layers):
        for p in ("weight", "bias"):
            out[f"blocks.{i}.attn.qkv.{p}"] = torch.cat([e[f"layer.{i}.attention.{q}.{p}"]
                                                         for q in ("query", "key", "value")], 0)
    return out


def main():
    os.makedirs(STAGE, exist_ok=True)
    t0 = time.time()
    record = {"reason": "Meta CDN TLS certificate expired 2026-11-11 23:59:59 UTC; Meta publishes V-JEPA 2.1 "
                        "nowhere else. Third-party re-upload accepted only after cross-verification (user-approved).",
              "official_source_url": "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt",
              "checkpoint_key": "ema_encoder", "checks": {}}

    print("1) canary: mnmly ViT-B vs our Meta-fetched ViT-B", flush=True)
    p_canary, rec_c = fetch(*CANARY)
    ours_b = os.path.join(MODELS, "vjepa2_1-vitb-384", "encoder.safetensors")
    same(load(p_canary), load(ours_b), "canary ViT-B (mnmly == Meta, fetched under valid TLS)")
    record["checks"]["canary_vitb"] = {**rec_c, "compared_with": "models/vjepa2_1-vitb-384/encoder.safetensors",
                                       "compared_sha256": sha256_file(ours_b), "result": "bit-identical"}

    print("2) primary ViT-L (mnmly)", flush=True)
    p_l, rec_l = fetch(*PRIMARY)
    L = load(p_l)
    print("3) independent ViT-L (apiantonio, Transformers layout)", flush=True)
    p_l2, rec_l2 = fetch(*SECOND)
    same(L, official_from_transformers(load(p_l2)), "ViT-L (mnmly == apiantonio after q/k/v -> qkv mapping)")
    record["checks"]["primary_vitl"] = rec_l
    record["checks"]["independent_vitl"] = {**rec_l2, "result": "all 302 encoder tensors equal after mapping"}

    print("4) architecture: strict load into vendored vit_large", flush=True)
    from third_party.vjepa2_1 import vision_transformer as vt

    enc = vt.vit_large(img_size=(384, 384), num_frames=64, tubelet_size=2, patch_size=16, use_sdpa=True,
                       use_silu=False, wide_silu=True, uniform_power=False, use_rope=True,
                       img_temporal_dim_size=1, interpolate_rope=True)
    res = enc.load_state_dict(L, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    n_blocks = len(enc.blocks)
    assert (len(L), n_blocks, enc.embed_dim) == (302, 24, 1024), (len(L), n_blocks, enc.embed_dim)
    record["checks"]["architecture"] = {"tensors": len(L), "blocks": n_blocks, "embed_dim": enc.embed_dim,
                                        "heads": enc.blocks[0].attn.num_heads, "strict_load": True}

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "encoder.safetensors")
    shutil.copyfile(p_l, out)
    man = {"source": "huggingface re-upload, cross-verified (see verification)", "source_url":
           f"https://huggingface.co/{PRIMARY[0]}/resolve/{rec_l['revision']}/{PRIMARY[2]}",
           "official_source_url": record["official_source_url"], "checkpoint_key": "ema_encoder",
           "num_tensors": len(L), "num_params": int(sum(v.numel() for v in L.values())),
           "output_sha256": sha256_file(out), "verification": record, "seconds": round(time.time() - t0, 1),
           "official_reverification": "pending: Meta CDN certificate expired"}
    json.dump(man, open(os.path.join(OUT_DIR, "manifest.json"), "w"), indent=2)
    src = os.path.join(MODELS, "MODEL_SOURCES.json")
    allsrc = json.load(open(src))
    allsrc["vjepa2_1-vitl-384"] = {k: v for k, v in man.items() if k != "verification"} | {"verification_checks":
                                                                                         list(record["checks"])}
    json.dump(allsrc, open(src, "w"), indent=2)
    print(f"ACCEPTED: {out} ({man['num_params']/1e6:.1f}M params, sha256 {man['output_sha256'][:16]}...)", flush=True)


if __name__ == "__main__":
    main()
