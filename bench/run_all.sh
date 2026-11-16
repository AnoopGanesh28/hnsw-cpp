#!/usr/bin/env bash
# The full M5b benchmark run on SIFT1M: builds every binary first, then runs
# the timed stages, writing JSON/text under bench/results/ with fixed names.
#
#   bench/run_all.sh [STAGE ...]     # default: build s0 s1 s2 s3 s4 s5 s6 s7
#
#   build  compile everything: the bench tools, the 8 ablation variants of
#          bench_search, the profile build, and the M2 layout baseline
#   s0     environment record (embeds env_manual.txt) and ground-truth check
#   s1     index builds, ours and hnswlib (3 runs each, one process per system)
#   s2     main recall/QPS sweep, threads 1 (pinned), 4 and 8
#   s3     warm-cache load times; prints the cold-cache command (needs sudo)
#   s4     ablation: {epoch, heaps, prefetch} x {scalar, AVX2} at two ef points
#   s5     layout ablation against the M2 baseline
#   s6     perf profile of ours and hnswlib
#   s7     repeat of the headline single-thread points
#
# Every stage skips outputs that already exist, so an interrupted run resumes
# where it stopped; delete a file to redo it. Timed stages refuse to start
# without bench/results/env_manual.txt, which every result file embeds.
# Do not run builds, tests or other heavy work while timed stages run.
#
# Settings (environment variables; the defaults are the M5b run):
#   HNSW_BENCH_RESULTS     results directory            (bench/results)
#   HNSW_BENCH_ENV_MANUAL  hand-written notes file      ($RESULTS/env_manual.txt)
#   HNSW_BENCH_PIN_CPU     logical CPU for pinned runs  (2)
#   HNSW_BENCH_N           base vectors to use          (all; a prefix is dev only)
#   HNSW_BENCH_EFS, HNSW_BENCH_SWEEP_REPS, HNSW_BENCH_BUILD_RUNS,
#   HNSW_BENCH_ABL_REPS, HNSW_BENCH_LAYOUT_REPS, HNSW_BENCH_LAYOUT_PROBE_N,
#   HNSW_BENCH_LAYOUT_SUBSET_N, HNSW_BENCH_LAYOUT_LIMIT_S   (see below)
#   HNSW_M2_WORKTREE       worktree of m2-scalar-baseline ($HOME/hnsw-m2-baseline)
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

RESULTS=${HNSW_BENCH_RESULTS:-bench/results}
ENV_MANUAL=${HNSW_BENCH_ENV_MANUAL:-$RESULTS/env_manual.txt}
PIN_CPU=${HNSW_BENCH_PIN_CPU:-2}
N=${HNSW_BENCH_N:-all}
EFS=${HNSW_BENCH_EFS:-10,12,14,16,18,20,24,28,32,40,48,56,64,80,96,112,128,160,200,256,320,400,500,640,800}
SWEEP_REPS=${HNSW_BENCH_SWEEP_REPS:-7}
BUILD_RUNS=${HNSW_BENCH_BUILD_RUNS:-3}
ABL_REPS=${HNSW_BENCH_ABL_REPS:-5}
LAYOUT_REPS=${HNSW_BENCH_LAYOUT_REPS:-7}
LAYOUT_PROBE_N=${HNSW_BENCH_LAYOUT_PROBE_N:-100000}
LAYOUT_SUBSET_N=${HNSW_BENCH_LAYOUT_SUBSET_N:-200000}
LAYOUT_LIMIT_S=${HNSW_BENCH_LAYOUT_LIMIT_S:-5400}
M2_WT=${HNSW_M2_WORKTREE:-$HOME/hnsw-m2-baseline}

DATA=bench/data/sift
BASE=$DATA/sift_base.fvecs
QUERIES=$DATA/sift_query.fvecs
INDEX=bench/data/index
B=build-bench/bench
PROF=build-prof/bench/bench_search
M2BIN=build-m2/bench_m2
PY="python3 -I $ROOT/bench/lib/benchlib.py"
PIN="taskset -c $PIN_CPU"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "error: $*"; exit 1; }

# Ground truth and index names for the configured base size.
if [[ $N == all ]]; then
  GT=$DATA/sift_groundtruth.ivecs
  NFLAG=()
  SUFFIX=""
else
  GT=$INDEX/gt_n$N.ivecs
  NFLAG=(--n "$N")
  SUFFIX="_n$N"
fi
OURS=$INDEX/ours$SUFFIX.hnsw
HNSWLIB=$INDEX/hnswlib$SUFFIX.idx

# Runs a stage, appending its wall time to $RESULTS/stage_times.tsv.
stage() {
  local name=$1
  shift
  local t0 t1
  t0=$(date +%s)
  log "stage $name: start"
  "$@"
  t1=$(date +%s)
  printf '%s\t%s\t%s\t%s\n' "$name" "$(date -u -d "@$t0" +%FT%TZ)" "$(date -u -d "@$t1" +%FT%TZ)" \
    "$((t1 - t0))" >> "$RESULTS/stage_times.tsv"
  log "stage $name: done in $((t1 - t0)) s"
}

# ---------------------------------------------------------------- build
variant_dir() { echo "build-abl/e$1h$2p$3"; }

do_build() {
  cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DHNSW_BUILD_BENCH=ON > /dev/null
  cmake --build build-bench -j "$(nproc)"
  local e h p d
  for e in 0 1; do for h in 0 1; do for p in 0 1; do
    d=$(variant_dir $e $h $p)
    cmake -S . -B "$d" -DCMAKE_BUILD_TYPE=Release -DHNSW_BUILD_BENCH=ON \
      -DHNSW_OPT_EPOCH_VISITED=$e -DHNSW_OPT_ALLOC_FREE_HEAPS=$h -DHNSW_OPT_PREFETCH=$p > /dev/null
    cmake --build "$d" -j "$(nproc)" --target bench_search
  done; done; done
  # Profile build: the release flags plus -g.
  cmake -S . -B build-prof -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-g \
    -DHNSW_BUILD_BENCH=ON > /dev/null
  cmake --build build-prof -j "$(nproc)" --target bench_search
  bench/m2_baseline/build.sh "$M2_WT" build-m2
  if [[ -n $(git status --porcelain --untracked-files=no) ]]; then
    log "warning: the working tree has uncommitted changes; results record it as dirty"
  fi
}

require_env_manual() {
  [[ -s $ENV_MANUAL ]] || die "$ENV_MANUAL is missing or empty: refusing to start timed stages"
  HNSW_BENCH_ENV_MANUAL=$(realpath "$ENV_MANUAL")
  export HNSW_BENCH_ENV_MANUAL
  local bin
  for bin in $B/bench_build $B/bench_search $B/bench_loadtime $B/bench_gt $PROF $M2BIN \
             $(for d in build-abl/*/; do echo "$d/bench/bench_search"; done); do
    [[ -x $bin ]] || die "$bin is not built: run the build stage first"
  done
  [[ $(ls -d build-abl/*/ | wc -l) == 8 ]] || die "expected 8 ablation variant builds"
}

# ---------------------------------------------------------------- s0
do_s0() {
  mkdir -p "$INDEX"
  [[ -f $RESULTS/environment.json ]] ||
    $PY env --out "$RESULTS/environment.json" --manual "$HNSW_BENCH_ENV_MANUAL" \
      --pin-cpu "$PIN_CPU" --n "$N" --m2-worktree "$M2_WT"
  if [[ $N == all ]]; then
    [[ -f $RESULTS/gt_verify.json ]] ||
      $B/bench_gt verify --base "$BASE" --queries "$QUERIES" --gt "$GT" \
        --json "$RESULTS/gt_verify.json"
  else
    [[ -f $GT ]] || $B/bench_gt subset --base "$BASE" --queries "$QUERIES" --n "$N" --out "$GT"
  fi
}

# ---------------------------------------------------------------- s1
do_s1() {
  local sys idx
  for sys in ours hnswlib; do
    idx=$OURS
    [[ $sys == hnswlib ]] && idx=$HNSWLIB
    if [[ -f $RESULTS/build_$sys.json && -f $idx ]]; then
      log "s1: build_$sys.json and its index exist, skipping"
      continue
    fi
    $B/bench_build --system $sys --data "$BASE" "${NFLAG[@]}" --M 16 --efc 200 \
      --runs "$BUILD_RUNS" --out "$idx" --json "$RESULTS/build_$sys.json"
  done
}

# ---------------------------------------------------------------- s2
sweep() {  # sweep OUT THREADS [launcher...]
  local out=$1 threads=$2
  shift 2
  if [[ -f $out ]]; then
    log "s2: $out exists, skipping"
    return
  fi
  "$@" $B/bench_search --ours "$OURS" --hnswlib "$HNSWLIB" --queries "$QUERIES" --gt "$GT" \
    --efs "$EFS" --threads "$threads" --reps "$SWEEP_REPS" --load-mode mmap --json "$out"
}

do_s2() {
  sweep "$RESULTS/sweep_t1.json" 1 $PIN
  sweep "$RESULTS/sweep_t4.json" 4
  sweep "$RESULTS/sweep_t8.json" 8
  $PY spread --threshold 0.15 --out "$RESULTS/sweep_high_spread.json" \
    "$RESULTS"/sweep_t1.json "$RESULTS"/sweep_t4.json "$RESULTS"/sweep_t8.json
}

# ---------------------------------------------------------------- s3
do_s3() {
  [[ -f $RESULTS/loadtime_warm.json ]] ||
    $B/bench_loadtime --ours "$OURS" --hnswlib "$HNSWLIB" --queries "$QUERIES" \
      --json "$RESULTS/loadtime_warm.json"
  local cmd
  cmd="HNSW_BENCH_ENV_MANUAL=$HNSW_BENCH_ENV_MANUAL bench/run_cold_loadtime.sh build-bench"
  cmd+=" $OURS $HNSWLIB $QUERIES $RESULTS/loadtime_cold.json"
  printf '%s\n' "$cmd" > "$RESULTS/cold_loadtime_command.txt"
  log "s3: cold-cache load time is not run here (needs sudo). From $ROOT run:"
  printf '\n  %s\n\n' "$cmd" >&2
  log "s3: it calls: sudo -v, then 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches' before every load"
}

# ---------------------------------------------------------------- s4
ABL=$RESULTS/ablation
COMBOS=()
for e in 0 1; do for h in 0 1; do for p in 0 1; do for k in scalar avx2; do
  COMBOS+=("$e$h$p$k")
done; done; done; done

ablation_efs() { $PY get --file "$RESULTS/ablation_efs.json" --key efs_csv; }

do_s4() {
  mkdir -p "$ABL"
  [[ -f $RESULTS/ablation_efs.json ]] ||
    $PY pick-efs --sweep "$RESULTS/sweep_t1.json" --system ours --targets 0.95,0.99 \
      --out "$RESULTS/ablation_efs.json"
  local efs r i j c e h p k kflag out
  efs=$(ablation_efs)
  for ((r = 1; r <= ABL_REPS; r++)); do
    # Round robin, starting one combination later on every repetition.
    for ((i = 0; i < ${#COMBOS[@]}; i++)); do
      j=$(((i + r - 1) % ${#COMBOS[@]}))
      c=${COMBOS[$j]}
      e=${c:0:1} h=${c:1:1} p=${c:2:1} k=${c:3}
      out=$ABL/abl_e${e}h${h}p${p}_${k}_r$r.json
      [[ -f $out ]] && continue
      kflag=best
      [[ $k == scalar ]] && kflag=scalar
      $PIN "$(variant_dir $e $h $p)/bench/bench_search" --ours "$OURS" --systems ours \
        --kernel $kflag --queries "$QUERIES" --gt "$GT" --efs "$efs" --threads 1 --reps 1 \
        --load-mode mmap --json "$out"
    done
  done
  $PY check-ablation --dir "$ABL" --reps "$ABL_REPS" --out "$RESULTS/ablation_check.json"
}

# ---------------------------------------------------------------- s5
LAY=$RESULTS/layout

do_s5() {
  mkdir -p "$LAY"
  [[ -f $LAY/m2_build_probe.json ]] ||
    $M2BIN --base "$BASE" --n "$LAYOUT_PROBE_N" --build-only --json "$LAY/m2_build_probe.json"
  [[ -f $LAY/projection.json ]] ||
    $PY project-layout --probe "$LAY/m2_build_probe.json" --base "$BASE" --n "$N" \
      --limit-seconds "$LAYOUT_LIMIT_S" --subset-n "$LAYOUT_SUBSET_N" --out "$LAY/projection.json"
  local ln lflag gt idx efs
  ln=$($PY get --file "$LAY/projection.json" --key layout_n)
  if [[ $($PY get --file "$LAY/projection.json" --key subset) == true ]]; then
    lflag=(--n "$ln")
    gt=$INDEX/gt_n$ln.ivecs
    idx=$INDEX/ours_scalar_n$ln.hnsw
    [[ -f $gt ]] || $B/bench_gt subset --base "$BASE" --queries "$QUERIES" --n "$ln" --out "$gt"
  else
    lflag=("${NFLAG[@]}")
    gt=$GT
    idx=$INDEX/ours_scalar$SUFFIX.hnsw
  fi
  # Ours built with the scalar kernel: the graph M2 builds, in the current layout.
  [[ -f $LAY/build_ours_scalar.json && -f $idx ]] ||
    $B/bench_build --system ours --kernel scalar --data "$BASE" "${lflag[@]}" --runs 1 \
      --out "$idx" --json "$LAY/build_ours_scalar.json"
  if [[ $($PY get --file "$LAY/projection.json" --key subset) == true ]]; then
    # Recall targets at the subset size: smallest ef of the grid reaching each
    # target with the all-off scalar variant (same results as M2, checked below).
    [[ -f $LAY/subset_sweep.json ]] ||
      $PIN "$(variant_dir 0 0 0)/bench/bench_search" --ours "$idx" --systems ours \
        --kernel scalar --queries "$QUERIES" --gt "$gt" --efs "$EFS" --threads 1 --reps 1 \
        --json "$LAY/subset_sweep.json"
    [[ -f $LAY/layout_efs.json ]] ||
      $PY pick-efs --sweep "$LAY/subset_sweep.json" --system ours --targets 0.95,0.99 \
        --out "$LAY/layout_efs.json"
  else
    [[ -f $LAY/layout_efs.json ]] || cp "$RESULTS/ablation_efs.json" "$LAY/layout_efs.json"
  fi
  efs=$($PY get --file "$LAY/layout_efs.json" --key efs_csv)
  if [[ ! -f $LAY/m2.json ]]; then
    rm -f "$LAY"/alloff_bracket_*.json  # brackets belong to one M2 run
    local hook
    hook="$ROOT/bench/lib/layout_bracket.sh $PIN_CPU $(variant_dir 0 0 0)/bench/bench_search"
    hook+=" $idx $QUERIES $gt $efs $LAY"
    $PIN $M2BIN --base "$BASE" "${lflag[@]}" --queries "$QUERIES" --gt "$gt" --efs "$efs" \
      --k 10 --reps "$LAYOUT_REPS" --between "$hook" --json "$LAY/m2.json.tmp"
    mv "$LAY/m2.json.tmp" "$LAY/m2.json"
  fi
  $PY check-layout --m2 "$LAY/m2.json" --dir "$LAY" --reps "$LAYOUT_REPS" \
    --out "$RESULTS/layout_check.json"
}

# ---------------------------------------------------------------- s6
find_perf() {
  local p
  for p in perf /usr/lib/linux-tools/*/perf /usr/lib/linux-tools-*/perf; do
    if command -v "$p" > /dev/null 2>&1 && "$p" stat -e cpu-clock true > /dev/null 2>&1; then
      command -v "$p"
      return 0
    fi
  done
  return 1
}

do_s6() {
  local out=$RESULTS/perf
  mkdir -p "$out" bench/data/perf
  local perf
  if ! perf=$(find_perf); then
    {
      echo "perf: not available"
      for p in perf /usr/lib/linux-tools/*/perf /usr/lib/linux-tools-*/perf; do
        echo "--- $p stat -e cpu-clock true"
        "$p" stat -e cpu-clock true 2>&1 || true
      done
    } > "$out/perf_unavailable.txt"
    log "s6: perf not available; see $out/perf_unavailable.txt"
    return 0
  fi
  local ef sys
  ef=$($PY get --file "$RESULTS/ablation_efs.json" --key ef_0.95)
  {
    echo "perf: $perf"
    "$perf" --version
    echo "binary: $PROF"
    echo "pinned cpu: $PIN_CPU"
    echo "ef: $ef (smallest ef with ours recall@10 >= 0.95 in sweep_t1.json)"
    echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
  } > "$out/perf_info.txt"
  for sys in ours hnswlib; do
    local args=(--systems $sys --queries "$QUERIES" --gt "$GT" --efs "$ef" --threads 1
                --reps 10)
    if [[ $sys == ours ]]; then args+=(--ours "$OURS" --validate shallow); else args+=(--hnswlib "$HNSWLIB"); fi
    if [[ ! -f $out/report_$sys.txt ]]; then
      $PIN "$perf" record -e cpu-clock -o "bench/data/perf/$sys.data" -- \
        $PROF "${args[@]}" --json "$out/profile_run_$sys.json" 2> "$out/record_$sys.log"
      "$perf" report --stdio --no-children --sort dso,symbol -i "bench/data/perf/$sys.data" \
        2> /dev/null | $PY head-report --entries 25 > "$out/report_$sys.txt"
    fi
    if [[ ! -f $out/stat_$sys.txt ]]; then
      $PIN "$perf" stat -e task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses \
        -o "$out/stat_$sys.txt" -- $PROF "${args[@]}" --json "$out/stat_run_$sys.json" 2> /dev/null
    fi
  done
}

# ---------------------------------------------------------------- s7
do_s7() {
  [[ -f $RESULTS/repeat_t1.json ]] && { log "s7: repeat_t1.json exists, skipping"; return; }
  $PIN $B/bench_search --ours "$OURS" --hnswlib "$HNSWLIB" --queries "$QUERIES" --gt "$GT" \
    --efs "$(ablation_efs)" --threads 1 --reps "$SWEEP_REPS" --load-mode mmap \
    --json "$RESULTS/repeat_t1.json"
}

# ---------------------------------------------------------------- main
main() {
  local stages=("$@")
  [[ ${#stages[@]} -gt 0 ]] || stages=(build s0 s1 s2 s3 s4 s5 s6 s7)
  mkdir -p "$RESULTS"
  local s
  for s in "${stages[@]}"; do
    case $s in
      build) stage build do_build ;;
      s[0-7])
        require_env_manual
        stage "$s" "do_$s" ;;
      *) die "unknown stage '$s'" ;;
    esac
  done
}

main "$@"
