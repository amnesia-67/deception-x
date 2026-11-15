#!/usr/bin/env python3
"""Compare the accepted ViT-L encoder (models/vjepa2_1-vitl-384) with Meta's OFFICIAL checkpoint, tensor by
tensor, once dl.fbaipublicfiles.com serves a valid TLS certificate again. Range-fetches only the ema_encoder
subtree (CRC-checked, safe unpickler) into memory; writes the result into the local manifest.

  bash scripts/run.sh scripts/verify_vitl_official.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fetch_models import MODELS, RANGE_MODELS, RangeTorchCheckpoint  # noqa: E402


def main():
    import torch
    from safetensors.torch import load_file

    spec = RANGE_MODELS["vjepa2_1-vitl-384"]
    d = os.path.join(MODELS, "vjepa2_1-vitl-384")
    local = load_file(os.path.join(d, "encoder.safetensors"))
    official = {}
    for k, v in RangeTorchCheckpoint(spec["url"]).load_subtree(spec["key"]).items():
        for p in spec["strip_prefixes"]:
            k = k.replace(p, "")
        official[k] = v
    if set(local) != set(official):
        raise SystemExit(f"MISMATCH key sets: {sorted(set(local) ^ set(official))[:5]}")
    bad = [k for k in local if not torch.equal(local[k], official[k].to(local[k].dtype)) or local[k].dtype != official[k].dtype]
    man_p = os.path.join(d, "manifest.json")
    man = json.load(open(man_p))
    man["official_reverification"] = "bit-identical to Meta's official checkpoint" if not bad else f"MISMATCH {bad[:5]}"
    man["official_reverification_detail"] = {
        "url": spec["url"], "checkpoint_key": spec["key"], "tensors_compared": len(local),
        "method": "range-fetched ema_encoder subtree under verified TLS, CRC-checked, compared tensor by tensor "
                  "(torch.equal and dtype) with models/vjepa2_1-vitl-384/encoder.safetensors",
        "date_utc": time.strftime("%Y-%m-%d", time.gmtime()), "mismatched_tensors": len(bad)}
    json.dump(man, open(man_p, "w"), indent=2)
    src = os.path.join(MODELS, "MODEL_SOURCES.json")
    allsrc = json.load(open(src))
    allsrc["vjepa2_1-vitl-384"]["official_reverification"] = man["official_reverification"]
    allsrc["vjepa2_1-vitl-384"]["official_reverification_detail"] = man["official_reverification_detail"]
    json.dump(allsrc, open(src, "w"), indent=2)
    print(man["official_reverification"])
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
