#!/usr/bin/env bash
# Regenerates tests/golden_m2.h: builds the M2 scalar baseline (git tag
# m2-scalar-baseline) in a temporary worktree with this project's flags
# (Release, -O3 -march=native, -ffp-contract=off) and records hashes of its
# graphs and search results for the configs in tests/golden_common.h.
set -euo pipefail

repo=$(git rev-parse --show-toplevel)
tmp=$(mktemp -d)
cleanup() {
  git -C "$repo" worktree remove --force "$tmp/m2" 2>/dev/null || true
  rm -rf "$tmp"
}
trap cleanup EXIT

git -C "$repo" worktree add --quiet --detach "$tmp/m2" m2-scalar-baseline
cmake -S "$tmp/m2" -B "$tmp/build" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS=-ffp-contract=off > /dev/null
cmake --build "$tmp/build" --target hnsw -j > /dev/null
"${CXX:-g++}" -std=c++17 -O3 -march=native -ffp-contract=off -Wall -Wextra \
    -I"$tmp/m2/include" "$repo/tests/golden_gen.cpp" "$tmp/build/libhnsw.a" \
    -o "$tmp/golden_gen"
"$tmp/golden_gen" > "$repo/tests/golden_m2.h"
echo "wrote $repo/tests/golden_m2.h"
