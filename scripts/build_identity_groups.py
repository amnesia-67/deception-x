#!/usr/bin/env python3
"""Write data/rlt_identity_groups.json: one group id per person in Real-Life Trials.

The dataset ships no speaker ids. The merges come from reviewing the face
contact sheets (data/review/contact_sheet_*.jpg) by eye. When two clips were
plausibly the same person or the same recording session, they were merged:
over-merging only makes the cross-validation stricter, under-merging would leak
identity across train and test. Every clip not listed is its own group.

The merge list describes the people in the videos, so it is not published. It is
read from data/rlt_identity_merges.local.json ({"<label>": ["trial_lie_001", ...]};
gitignored), made by the same contact-sheet review.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.rlt_data import GROUPS_PATH, index_clips  # noqa: E402

MERGES_PATH = os.path.join(ROOT, "data", "rlt_identity_merges.local.json")


def load_merges():
    if not os.path.exists(MERGES_PATH):
        raise SystemExit(f"{MERGES_PATH} not found: create it from the contact-sheet review (it is never published)")
    return json.load(open(MERGES_PATH, encoding="utf-8"))


def main():
    MERGES = load_merges()
    clips = [c.clip_id for c in index_clips()]
    seen = {}
    for name, members in MERGES.items():
        for m in members:
            if m not in clips:
                raise SystemExit(f"unknown clip {m} in {name}")
            if m in seen:
                raise SystemExit(f"{m} in both {seen[m]} and {name}")
            seen[m] = name
    clip_to_group, names, gid = {}, {}, 0
    for name, members in MERGES.items():
        names[gid] = name
        for m in members:
            clip_to_group[m] = gid
        gid += 1
    for c in clips:
        if c not in clip_to_group:
            names[gid] = f"single:{c}"
            clip_to_group[c] = gid
            gid += 1
    out = {
        "method": "manual visual review of face contact sheets; conservative merges",
        "n_groups": gid,
        "group_names": names,
        "clip_to_group": clip_to_group,
    }
    with open(GROUPS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    both = [n for g, n in names.items()
            if len({("lie" in c) for c, gg in clip_to_group.items() if gg == g}) == 2]
    print(f"{gid} identity groups over {len(clips)} clips; groups with both labels: {both}")


if __name__ == "__main__":
    main()
