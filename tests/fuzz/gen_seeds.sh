#!/usr/bin/env bash
# Regenerates the committed seed corpora in tests/fuzz/seeds/ with make_seeds.
#
# Usage: tests/fuzz/gen_seeds.sh <build-dir>   (any configured build of this repo)
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 <build-dir>" >&2; exit 2; }
build=$1
here=$(cd "$(dirname "$0")" && pwd)

cmake --build "$build" --target make_seeds >/dev/null
for t in loader fvecs; do
  rm -rf "${here:?}/seeds/$t"
  mkdir -p "$here/seeds/$t"
done
"$build/tests/fuzz/make_seeds" "$here/seeds/loader" "$here/seeds/fvecs"
for t in loader fvecs; do
  echo "seeds/$t: $(ls "$here/seeds/$t" | wc -l) files, $(du -sb "$here/seeds/$t" | cut -f1) bytes"
done
