#!/usr/bin/env bash
# Called by bench_m2 (--between) before its first and after every repetition:
# measures the all-off scalar variant once (1 warm-up + 1 timed pass per ef,
# pinned) into DIR/alloff_bracket_I.json. See bench/run_all.sh, stage s5.
#
# Usage: layout_bracket.sh PIN_CPU BENCH_SEARCH INDEX QUERIES GT EFS DIR I
set -euo pipefail
[[ $# -eq 8 ]] || { echo "usage: $0 PIN_CPU BENCH_SEARCH INDEX QUERIES GT EFS DIR I" >&2; exit 2; }
pin=$1 bin=$2 index=$3 queries=$4 gt=$5 efs=$6 dir=$7 i=$8
taskset -c "$pin" "$bin" --ours "$index" --systems ours --kernel scalar --queries "$queries" \
  --gt "$gt" --efs "$efs" --threads 1 --reps 1 --load-mode mmap \
  --json "$dir/alloff_bracket_$i.json" 2> /dev/null
