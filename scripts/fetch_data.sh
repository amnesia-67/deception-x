#!/usr/bin/env bash
# Download Real-Life Trials (Perez-Rosas et al., ICMI 2015) from the official University of Michigan page
# and verify it against the SHA256 of the copy used for every result in this repo.
#   bash scripts/fetch_data.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/raw/real_life_trials"
ZIP="$DEST/RealLifeDeceptionDetection.2016.zip"
URL="http://web.eecs.umich.edu/~mihalcea/downloads/RealLifeDeceptionDetection.2016.zip"
SHA256="dfa2476ee4929214a1985e1f1f755ebb2115c17e67588c2e832a16ce3a9921dc"
SIZE=248760721

mkdir -p "$DEST"
if [ ! -f "$ZIP" ]; then
  curl -fL --retry 3 -o "$ZIP.part" "$URL"
  mv "$ZIP.part" "$ZIP"
fi
[ "$(stat -c %s "$ZIP")" = "$SIZE" ] || { echo "size mismatch: $ZIP" >&2; exit 1; }
echo "$SHA256  $ZIP" | sha256sum -c -
[ -d "$DEST/Real-life_Deception_Detection_2016" ] || (cd "$DEST" && unzip -q "$ZIP")
echo "Real-Life Trials ready in $DEST/Real-life_Deception_Detection_2016"
