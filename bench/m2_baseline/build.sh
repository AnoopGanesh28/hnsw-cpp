#!/usr/bin/env bash
# Builds bench_m2 against the M2 scalar baseline (tag m2-scalar-baseline).
# Creates the git worktree WORKTREE (outside this repository) if it does not
# exist, checks that it is at the tag's commit and clean, and compiles every
# M2 source with -O3 -march=native -ffp-contract=off -DNDEBUG: the scalar
# kernel then rounds exactly like the current library's (whose
# src/distance.cpp uses -ffp-contract=off), so M2 builds the same graph as
# ours built with the scalar kernel and returns the same results.
#
# Usage: bench/m2_baseline/build.sh WORKTREE OUT_DIR
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 WORKTREE OUT_DIR" >&2; exit 2; }
wt=$1 out=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
tag=m2-scalar-baseline
want=$(git -C "$root" rev-parse "$tag^{commit}")
if [[ ! -d $wt ]]; then
  git -C "$root" worktree add --detach "$wt" "$tag" > /dev/null
fi
have=$(git -C "$wt" rev-parse HEAD)
[[ $have == "$want" ]] || { echo "error: $wt is at $have, not $tag ($want)" >&2; exit 1; }
[[ -z $(git -C "$wt" status --porcelain) ]] || { echo "error: $wt has local changes" >&2; exit 1; }
mkdir -p "$out"
cxx=${CXX:-g++}
flags="-std=c++17 -O3 -march=native -ffp-contract=off -DNDEBUG -Wall -Wextra"
# shellcheck disable=SC2086
"$cxx" $flags -I"$wt/include" -DBENCH_M2_FLAGS="\"$cxx $flags\"" -DBENCH_M2_COMMIT="\"$want\"" \
  "$root/bench/m2_baseline/bench_m2.cpp" "$wt"/src/*.cpp -o "$out/bench_m2.tmp"
mv "$out/bench_m2.tmp" "$out/bench_m2"
