#!/usr/bin/env bash
# The canonical full configuration: a Release build with the benchmark tools and
# the Python extension ON, and the whole ctest suite run serially and verbosely
# so every test's own output (check counts, the unittest summary) is saved to
# docs/evidence/ctest_<compiler>_release.txt.
#
# Usage: scripts/ctest_full.sh <gcc|clang>
#
# Needs the hnswlib baseline (bench/get_hnswlib.sh) and pybind11; it fails
# instead of silently building a smaller configuration.
set -euo pipefail

usage() { echo "usage: $0 <gcc|clang>" >&2; exit 2; }
[[ $# -eq 1 ]] || usage
case "$1" in
  gcc) cxx=g++ ;;
  clang) cxx=clang++ ;;
  *) usage ;;
esac
compiler=$1

root=$(cd "$(dirname "$0")/.." && pwd)
build="$root/build-full-$compiler"
log="$root/docs/evidence/ctest_${compiler}_release.txt"
[[ -f "$root/bench/third_party/hnswlib/hnswlib/hnswlib.h" ]] \
  || { echo "hnswlib missing: run bench/get_hnswlib.sh" >&2; exit 1; }
python3 -c "import pybind11" 2>/dev/null || { echo "pybind11 for python3 missing" >&2; exit 1; }
mkdir -p "$root/docs/evidence"

rm -rf "$build"
cmake -S "$root" -B "$build" -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_BUILD_TYPE=Release \
  -DHNSW_BUILD_BENCH=ON -DHNSW_BUILD_PYTHON=ON >/dev/null
cmake --build "$build" -j "$(nproc)" >/dev/null

{
  echo "# scripts/ctest_full.sh $compiler"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit: $(git -C "$root" rev-parse HEAD)$(git -C "$root" diff --quiet HEAD -- . ':!docs/evidence' || echo ' (with uncommitted changes)')"
  echo "compiler: $($cxx --version | head -n 1)"
  echo "kernel: $(uname -sr)"
  echo "CMAKE_BUILD_TYPE=Release HNSW_SANITIZE=off HNSW_BUILD_BENCH=ON HNSW_BUILD_PYTHON=ON HNSW_BUILD_FUZZ=OFF"
  echo "compile flags (test_distance): $(grep -h '^CXX_FLAGS' "$build/CMakeFiles/test_distance.dir/flags.make" | cut -d= -f2-)"
  echo
} >"$log"

status=0
(cd "$build" && ctest -V) >>"$log" 2>&1 || status=$?
echo "ctest exit status: $status" >>"$log"
tail -n 4 "$log"
exit "$status"
