#!/usr/bin/env bash
# Builds the project with a sanitizer in a fresh build directory and runs the
# full ctest suite, saving the output to docs/evidence/ctest_<compiler>_<mode>.txt.
#
# Usage: scripts/sanitize.sh <gcc|clang> <address|undefined|address+undefined|thread>
#
# The benchmark tools are built (and their tests run) when the hnswlib baseline
# is present (bench/get_hnswlib.sh); the Python extension is built when pybind11
# is found. Python tests are not part of the sanitizer suite (the interpreter is
# not instrumented); see scripts/sanitize_python.sh.
set -euo pipefail

usage() {
  echo "usage: $0 <gcc|clang> <address|undefined|address+undefined|thread>" >&2
  exit 2
}
[[ $# -eq 2 ]] || usage
compiler=$1
mode=$2

case "$compiler" in
  gcc) cxx=g++ ;;
  clang) cxx=clang++ ;;
  *) usage ;;
esac
case "$mode" in
  address | undefined | thread) sanitize=$mode ;;
  address+undefined) sanitize=address,undefined ;;
  *) usage ;;
esac

root=$(cd "$(dirname "$0")/.." && pwd)
build="$root/build-san-$compiler-${mode/+/-}"
evidence="$root/docs/evidence"
log="$evidence/ctest_${compiler}_${mode}.txt"
mkdir -p "$evidence"

bench=OFF
[[ -f "$root/bench/third_party/hnswlib/hnswlib/hnswlib.h" ]] && bench=ON
python=OFF
python3 -c "import pybind11" 2>/dev/null && python=ON

# -O1 -g without NDEBUG: assertions stay on and stack traces stay readable.
rm -rf "$build"
cmake -S "$root" -B "$build" \
  -DCMAKE_CXX_COMPILER="$cxx" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O1 -g" \
  -DHNSW_SANITIZE="$sanitize" \
  -DHNSW_BUILD_BENCH="$bench" \
  -DHNSW_BUILD_PYTHON="$python" >/dev/null
cmake --build "$build" -j "$(nproc)" >/dev/null

export ASAN_OPTIONS=detect_leaks=1:abort_on_error=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1
export TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1

{
  echo "# scripts/sanitize.sh $compiler $mode"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit: $(git -C "$root" rev-parse HEAD)$(git -C "$root" diff --quiet HEAD -- . ':!docs/evidence' || echo ' (with uncommitted changes)')"
  echo "compiler: $($cxx --version | head -n 1)"
  echo "kernel: $(uname -sr)"
  echo "HNSW_SANITIZE=$sanitize HNSW_BUILD_BENCH=$bench HNSW_BUILD_PYTHON=$python"
  echo "compile flags (test_distance): $(grep -h '^CXX_FLAGS' "$build/CMakeFiles/test_distance.dir/flags.make" | cut -d= -f2-)"
  echo "ASAN_OPTIONS=$ASAN_OPTIONS"
  echo "UBSAN_OPTIONS=$UBSAN_OPTIONS"
  echo "TSAN_OPTIONS=$TSAN_OPTIONS"
  echo
} >"$log"

status=0
(cd "$build" && ctest --output-on-failure -j 4) >>"$log" 2>&1 || status=$?
echo "ctest exit status: $status" >>"$log"
tail -n 4 "$log"
exit "$status"
