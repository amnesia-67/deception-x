"""Participant pseudonyms for the SunHacks webcam pilot.

The pilot exports identify people by name in `participant_id`. Code, reports and manifests in this repository use
the pseudonyms P1..P5 instead. The mapping lives only on the machine that holds the data, in
data/participant_aliases.local.json ({"<participant_id>": "P1", ...}; gitignored, never published). Without that
file every id passes through unchanged, so the pilot scripts then need the export's own ids.
"""

from __future__ import annotations

import json
import os
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIASES_PATH = os.environ.get("SUNHACKS_PARTICIPANT_ALIASES",
                              os.path.join(ROOT, "data", "participant_aliases.local.json"))


def _load() -> Dict[str, str]:
    if not os.path.exists(ALIASES_PATH):
        return {}
    m = json.load(open(ALIASES_PATH, encoding="utf-8"))
    if len(set(m.values())) != len(m):
        raise ValueError(f"{ALIASES_PATH}: two participants share a pseudonym")
    return m


_ALIASES = _load()
_REAL = {v: k for k, v in _ALIASES.items()}


def alias(participant_id: str) -> str:
    """Pseudonym (P1..) for an export's participant_id; unchanged when no mapping is installed."""
    return _ALIASES.get(participant_id, participant_id)


def real(pseudonym: str) -> str:
    """The export's participant_id for a pseudonym (to select its rows or folds in the export's own files)."""
    return _REAL.get(pseudonym, pseudonym)
