#!/usr/bin/env bash
# Runs the Python tests and the quickstart example against the ASan-instrumented
# extension built by `scripts/sanitize.sh <compiler> address`, saving the output
# to docs/evidence/python_asan_<compiler>.txt.
#
# Usage: scripts/sanitize_python.sh <gcc|clang>
#
# The interpreter is not instrumented, so the ASan runtime is preloaded, and
# libstdc++ with it: Python does not link libstdc++, and ASan's __cxa_throw
# interceptor must find the real one when it initializes. Leak detection is off
# because the interpreter's own allocations at exit are not ours to judge.
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 <gcc|clang>" >&2; exit 2; }
case "$1" in
  gcc) runtime=$(g++ -print-file-name=libasan.so) ;;
  clang) runtime=$(clang++ -print-file-name=libclang_rt.asan-x86_64.so) ;;
  *) echo "usage: $0 <gcc|clang>" >&2; exit 2 ;;
esac
compiler=$1

root=$(cd "$(dirname "$0")/.." && pwd)
build="$root/build-san-$compiler-address"
log="$root/docs/evidence/python_asan_$compiler.txt"
module=$(ls "$build"/python/hnsw_cpp/_core*.so 2>/dev/null || true)
if [[ -z "$module" ]]; then
  echo "no extension in $build; run scripts/sanitize.sh $compiler address first" >&2
  exit 1
fi

export PYTHONPATH="$build/python"
export LD_PRELOAD="$runtime:$(g++ -print-file-name=libstdc++.so)"
export ASAN_OPTIONS=detect_leaks=0:abort_on_error=1

{
  echo "# scripts/sanitize_python.sh $compiler"
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit: $(git -C "$root" rev-parse HEAD)$(git -C "$root" diff --quiet HEAD -- . ':!docs/evidence' || echo ' (with uncommitted changes)')"
  echo "python: $(python3 --version)"
  echo "module: ${module#"$root"/}"
  echo "ASan report hooks in the module: $(nm -D "$module" | grep -c '__asan_report_')"
  echo "LD_PRELOAD=$LD_PRELOAD"
  echo "ASAN_OPTIONS=$ASAN_OPTIONS"
  echo
} >"$log"

status=0
python3 -m unittest discover -v -s "$root/python/tests" >>"$log" 2>&1 || status=$?
if [[ $status -eq 0 ]]; then
  python3 "$root/python/examples/quickstart.py" >>"$log" 2>&1 || status=$?
fi
echo "exit status: $status" >>"$log"
tail -n 3 "$log"
exit "$status"
