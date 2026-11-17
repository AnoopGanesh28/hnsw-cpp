#!/usr/bin/env bash
# Runs a libFuzzer target for a fixed wall time and saves the full log to
# docs/evidence/fuzz_<target>.log. If no crash was found, the corpus is then
# minimized (-merge=1) into tests/fuzz/corpus/<target>, keeping only inputs that
# add coverage beyond the committed seeds.
#
# Usage: scripts/run_fuzz.sh <fuzz-build-dir> <loader|fvecs> <seconds> [workers]
#
# The build dir must be configured with clang, -DHNSW_SANITIZE=address,undefined
# and -DHNSW_BUILD_FUZZ=ON. Workers run in libFuzzer's fork mode (-fork=N),
# which reports totals over all workers in one log.
set -euo pipefail

usage() { echo "usage: $0 <fuzz-build-dir> <loader|fvecs> <seconds> [workers]" >&2; exit 2; }
[[ $# -eq 3 || $# -eq 4 ]] || usage
build=$(cd "$1" && pwd)
target=$2
seconds=$3
workers=${4:-4}
[[ $target == loader || $target == fvecs ]] || usage

root=$(cd "$(dirname "$0")/.." && pwd)
fuzz="$root/tests/fuzz"
bin="$build/tests/fuzz/fuzz_$target"
work="$build/fuzz-run/$target"
log="$root/docs/evidence/fuzz_$target.log"
[[ -x $bin ]] || { echo "missing $bin (configure with -DHNSW_BUILD_FUZZ=ON)" >&2; exit 1; }

cmake --build "$build" --target "fuzz_$target" >/dev/null
rm -rf "$work"
mkdir -p "$work/corpus" "$work/artifacts" "$(dirname "$log")"

args=(-fork="$workers" -max_total_time="$seconds" -timeout=10 -dict="$fuzz/$target.dict"
      -artifact_prefix="$work/artifacts/")
{
  echo "# scripts/run_fuzz.sh $target $seconds $workers"
  echo "start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit: $(git -C "$root" rev-parse HEAD)$(git -C "$root" diff --quiet HEAD -- . ':!docs/evidence' ':!tests/fuzz/corpus' || echo ' (with uncommitted changes)')"
  echo "compiler: $(clang++ --version | head -n 1)"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) logical CPUs"
  echo "compile flags: $(grep -h '^CXX_FLAGS' "$build/CMakeFiles/hnsw.dir/flags.make" | cut -d= -f2-)"
  echo "command: fuzz_$target ${args[*]} <new corpus> seeds/$target corpus/$target"
  echo
} >"$log"

status=0
(cd "$work" && "$bin" "${args[@]}" corpus "$fuzz/seeds/$target" "$fuzz/corpus/$target") \
  >>"$log" 2>&1 || status=$?
{
  echo
  echo "end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "fuzzer exit status: $status"
  echo "crash artifacts: $(find "$work/artifacts" -type f | wc -l)"
  find "$work/artifacts" -type f -printf '  %f\n'
} >>"$log"
if [[ $status -ne 0 ]]; then
  echo "fuzzer failed (status $status); artifacts in $work/artifacts" >&2
  exit "$status"
fi

# Minimize: start from the seeds so only inputs that add coverage beyond them
# are written, then drop the seed copies.
merged="$work/merged"
mkdir -p "$merged"
cp "$fuzz/seeds/$target"/* "$merged/"
echo >>"$log"
echo "# merge: fuzz_$target -merge=1 <seeds> <new corpus> corpus/$target" >>"$log"
"$bin" -merge=1 "$merged" "$work/corpus" "$fuzz/corpus/$target" >>"$log" 2>&1
for f in "$fuzz/seeds/$target"/*; do rm -f "$merged/$(basename "$f")"; done
find "$fuzz/corpus/$target" -type f ! -name .gitkeep -delete
cp "$merged"/* "$fuzz/corpus/$target/" 2>/dev/null || true
files=$(find "$fuzz/corpus/$target" -type f ! -name .gitkeep | wc -l)
bytes=$(find "$fuzz/corpus/$target" -type f ! -name .gitkeep -printf '%s\n' | awk '{s += $1} END {print s + 0}')
echo "corpus after merge: $files files, $bytes bytes (beyond the seeds)" >>"$log"
tail -n 1 "$log"
