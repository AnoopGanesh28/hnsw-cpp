#!/usr/bin/env bash
# Downloads SIFT1M (INRIA TEXMEX corpus) into bench/data/sift/, keeps only the
# three files the benchmarks use, records their sizes and sha256 in MANIFEST,
# and validates them with the repository's own fvecs/ivecs readers.
#
# Usage: bench/get_sift.sh [BUILD_DIR]
#   BUILD_DIR  CMake build directory configured with -DHNSW_BUILD_BENCH=ON
#              (default: build-bench). Validation runs BUILD_DIR/bench/bench_gt.
#
# The HTTP mirror answers 403, so this uses the FTP URL. Interrupted downloads
# resume. If the download still fails after retries the script exits nonzero;
# it never substitutes another dataset (bench/make_synthetic.cpp is the
# explicit, separately labelled fallback).
set -euo pipefail

URL=ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
ATTEMPTS=5
FILES=(sift_base.fvecs sift_query.fvecs sift_groundtruth.ivecs)

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(dirname "$here")"
build_dir="${1:-$repo/build-bench}"
data="$here/data"
out="$data/sift"
tarball="$data/sift.tar.gz"
bench_gt="$build_dir/bench/bench_gt"

mkdir -p "$out"

if [[ -f "$tarball" ]] && gzip -t "$tarball" 2>/dev/null; then
  echo "using existing $tarball"
else
  ok=0
  for attempt in $(seq 1 "$ATTEMPTS"); do
    echo "downloading $URL (attempt $attempt of $ATTEMPTS)"
    # -C - resumes a partial file; the speed limit aborts a stalled transfer
    # so the next attempt can resume it.
    if curl --fail --location --continue-at - --retry 3 --retry-delay 10 \
        --connect-timeout 30 --speed-limit 10240 --speed-time 60 \
        --output "$tarball" "$URL" && gzip -t "$tarball" 2>/dev/null; then
      ok=1
      break
    fi
    sleep 10
  done
  if [[ "$ok" != 1 ]]; then
    echo "error: could not download SIFT1M from $URL after $ATTEMPTS attempts." >&2
    echo "Partial data (if any) is in $tarball; rerun to resume." >&2
    echo "No dataset was substituted." >&2
    exit 1
  fi
fi

tar -xzf "$tarball" -C "$out" --strip-components=1 "${FILES[@]/#/sift/}"

{
  echo "dataset: sift1m"
  echo "source: $URL"
  echo "archive_sha256: $(sha256sum "$tarball" | cut -d ' ' -f 1)"
  echo "manifest_written_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for f in "${FILES[@]}"; do
    echo "file: $f $(stat -c %s "$out/$f") $(sha256sum "$out/$f" | cut -d ' ' -f 1)"
  done
} > "$out/MANIFEST"
echo "wrote $out/MANIFEST"

if [[ ! -x "$bench_gt" ]]; then
  echo "error: $bench_gt not found; configure with -DHNSW_BUILD_BENCH=ON, build," >&2
  echo "and rerun this script to validate the files (the download is kept)." >&2
  exit 1
fi
"$bench_gt" check --base "$out/sift_base.fvecs" --queries "$out/sift_query.fvecs" \
  --gt "$out/sift_groundtruth.ivecs" --n 1000000 --nq 10000 --dim 128 --k 100
