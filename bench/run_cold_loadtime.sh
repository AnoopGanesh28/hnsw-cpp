#!/usr/bin/env bash
# Cold-page-cache variant of the load-time benchmark: runs bench_loadtime with
# `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` before every single load.
# Needs sudo; run it yourself (it asks for the password once, up front).
#
# Usage: bench/run_cold_loadtime.sh BUILD_DIR OURS_INDEX HNSWLIB_INDEX QUERIES OUT_JSON [REPS]
#
# Under WSL2 this drops the page cache of the Linux VM only. The VM's virtual
# disk is a file on the Windows host, which may still have it cached, so
# "cold" there means cold for Linux, not necessarily a read from the device.
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 BUILD_DIR OURS_INDEX HNSWLIB_INDEX QUERIES OUT_JSON [REPS]" >&2
  exit 2
fi
build_dir="$1" ours="$2" hnswlib="$3" queries="$4" out="$5" reps="${6:-5}"
tool="$build_dir/bench/bench_loadtime"
if [[ ! -x "$tool" ]]; then
  echo "error: $tool not found (configure with -DHNSW_BUILD_BENCH=ON and build)" >&2
  exit 1
fi

# Cache credentials now; the per-load command uses sudo -n and fails loudly
# (and so does the benchmark) if they expire during the run.
sudo -v

"$tool" --ours "$ours" --hnswlib "$hnswlib" --queries "$queries" --json "$out" \
  --reps "$reps" \
  --pre-load-command 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null'
