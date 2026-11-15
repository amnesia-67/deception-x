"""Real-Life Trials (Perez-Rosas et al., ICMI 2015) dataset index.

121 real courtroom clips (61 deceptive / 60 truthful) with human transcripts,
downloaded from the official University of Michigan page by
``scripts/fetch_data.sh``. Nothing here is synthetic.

Identity groups for speaker-independent cross-validation live in
``data/rlt_identity_groups.json`` (built by ``scripts/build_identity_groups.py``
and reviewed by eye). Clips of one person never appear in both train and test.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RLT_DIR = os.path.join(ROOT, "data", "raw", "real_life_trials", "Real-life_Deception_Detection_2016")
GROUPS_PATH = os.path.join(ROOT, "data", "rlt_identity_groups.json")
CACHE_DIR = os.environ.get("DECEPTIONX_CACHE", os.path.expanduser("~/deceptionx_cache"))


@dataclass
class ClipRecord:
    clip_id: str            # e.g. "trial_lie_001"
    video_path: str
    label: int              # 1 = deceptive, 0 = truthful
    transcript: str
    gestures: Dict[str, int] = field(default_factory=dict)  # human MUMIN codes (reference only)
    group: Optional[int] = None


def _read_text(path: str) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                return " ".join(f.read().split())
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("could not decode", path.encode(), 0, 1, "")


def load_gestures(root: str = RLT_DIR) -> Dict[str, Dict[str, int]]:
    path = os.path.join(root, "Annotation", "All_Gestures_Deceptive and Truthful.csv")
    out: Dict[str, Dict[str, int]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = row.pop("id").replace(".mp4", "").strip()
            row.pop("class", None)
            out[cid] = {k: int(v) for k, v in row.items() if v not in ("", None)}
    return out


def load_groups(path: str = GROUPS_PATH) -> Dict[str, int]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {cid: int(g) for cid, g in data["clip_to_group"].items()}


def index_clips(root: str = RLT_DIR, require_groups: bool = False) -> List[ClipRecord]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Real-Life Trials not found at {root}; run scripts/fetch_data.sh")
    gestures = load_gestures(root)
    groups = load_groups()
    recs: List[ClipRecord] = []
    for sub, label in (("Deceptive", 1), ("Truthful", 0)):
        vdir = os.path.join(root, "Clips", sub)
        for fn in sorted(os.listdir(vdir)):
            if not fn.endswith(".mp4") or fn.startswith("._"):
                continue
            cid = fn[:-4]
            tpath = os.path.join(root, "Transcription", sub, cid + ".txt")
            recs.append(ClipRecord(
                clip_id=cid,
                video_path=os.path.join(vdir, fn),
                label=label,
                transcript=_read_text(tpath) if os.path.exists(tpath) else "",
                gestures=gestures.get(cid, {}),
                group=groups.get(cid),
            ))
    if require_groups:
        missing = [r.clip_id for r in recs if r.group is None]
        if missing:
            raise RuntimeError(f"{len(missing)} clips lack an identity group: {missing[:5]}...")
    return recs
