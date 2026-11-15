# Exact source served on Node 2 port 5002

`server_revision` (in `ping`, `analyze` and `jepa_video` replies) names the commit the server runs, says whether any
tracked file differs from it, and hashes the served source tree and every served model manifest. Anyone can check it
against this repository's git objects.

## Record and verify

On Node 2, with the server running from a checkout of this repository:

```bash
bash scripts/run.sh scripts/served_source_record.py        # writes reports/served_5002/record.json from the live process
```

On any machine (no GPU, no WSL):

```bash
python scripts/verify_served_revision.py reports/served_5002/record.json   # ends with ALL CHECKS PASSED
```

Records stay local (`reports/served_5002/` is gitignored) because they include local paths and process details.

## What `server_revision` contains

| field | meaning |
|---|---|
| `commit` | the checked-out commit |
| `dirty`, `modified_tracked_files` | content-based: a tracked file differs from the commit (raw bytes, CRLF→LF bytes or git-lfs object). Windows git and WSL give the same answer |
| `source_tree_sha256` | tracked source files only (`main.py`, `run_v2_socket.py`, `*.py`/`*.sh`/`*.ps1` under `src/`, `scripts/`, `third_party/`), CRLF→LF; untracked ones are listed in `untracked_source_files` with their SHA256 |
| `artifacts_sha256` | SHA256 of the CRLF→LF bytes of each served manifest or model file, equal to its git blob on Windows, WSL and LF clones |
| `recorded_at_unix` | when it was taken (once, at startup) |

Base weights that are not in git (dasheng-base LFS object, emotion2vec, Qwen3, V-JEPA 2.1 ViT-L encoder) are bound by
SHA256 in `models/MODEL_SOURCES.json` and by the hashed manifests.
