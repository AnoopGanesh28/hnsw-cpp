# Benchmarks

Tools that measure this library against [hnswlib](https://github.com/nmslib/hnswlib)
on SIFT1M, following the method in [docs/TASK.md](../docs/TASK.md) ("Benchmark
methodology"). Every number reported in the top-level README must come from a JSON
file these tools wrote into `bench/results/`. This file describes the tools and the
method. It contains no results.

## Setup

```sh
bench/get_hnswlib.sh                                    # hnswlib v0.9.0 into bench/third_party/ (pinned commit)
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DHNSW_BUILD_BENCH=ON
cmake --build build-bench -j
bench/get_sift.sh build-bench                           # SIFT1M into bench/data/sift/, then validates it
```

- `HNSW_BUILD_BENCH` is `OFF` by default. With it off, nothing in `bench/` is built and
  the normal test suite is unchanged. With it on, configuring fails with a message if
  hnswlib is missing or checked out at a commit other than the one pinned in
  `get_hnswlib.sh`.
- `get_hnswlib.sh` checks the tag against the remote with `git ls-remote` before
  cloning, then checks the clone's commit. The pin (tag and commit) lives only in
  that script, and CMake reads it from there.
- `get_sift.sh` downloads `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` with
  curl. The HTTP mirror returns 403. The download resumes and retries. The script
  extracts only `sift_base.fvecs`, `sift_query.fvecs` and `sift_groundtruth.ivecs`,
  writes `bench/data/sift/MANIFEST` (sizes and sha256), and runs `bench_gt check`. That
  check uses the library's own fvecs/ivecs readers and expects a base of 1,000,000 x 128,
  queries of 10,000 x 128, a ground truth of 10,000 x 100, and ids in range. If the
  download fails, the script exits nonzero. It never substitutes another dataset.
- `bench/data/`, `bench/third_party/` and `bench/results_scratch/` are gitignored.

Every tool reads the `MANIFEST` in the directory of its data files. It records the
dataset name in its output, and it refuses a file whose size or sha256 differs from
its manifest entry.

### Synthetic fallback

Use this only if SIFT1M cannot be downloaded:

```sh
build-bench/bench/bench_make_synthetic --out bench/data/synthetic   # 1M x 128, 10k queries, top-100
```

The data is a seeded Gaussian mixture in the SIFT1M layout. Its ground truth is an
exact, multi-threaded brute force. Its MANIFEST names the dataset `synthetic`, so
every result file built from it says `"name": "synthetic"`. Source:
[make_synthetic.cpp](make_synthetic.cpp).

## Tools

All tools take `--flag value` pairs and reject unknown flags. Each writes one JSON
file, and each one's header comment lists every flag.

| Tool | What it does |
|---|---|
| `bench_gt check` | Shape and id-range validation of a dataset (used by `get_sift.sh`). |
| `bench_gt verify` | Recomputes the exact top-K for `--sample` queries (default 100, spread evenly) by multi-threaded brute force. It compares **distances** rank by rank with the distances of the provided ids, so tied neighbors may swap ids without counting as an error. Exits 1 on any mismatch. |
| `bench_gt subset` | Exact top-k of every query among the first `--n` base vectors, written as ivecs, for dev-scale runs. |
| `bench_build` | Builds one index for `--system ours\|hnswlib`, one system per process. Records build time per run, VmHWM after loading the data and at the end, and the index file size. |
| `bench_search` | Sweeps recall@k against QPS over `--efs` and `--threads` for both systems (method below). |
| `bench_loadtime` | Load time and the time to answer the first `--first` queries for five strategies: ours mmap / read, each with shallow / deep validation, and hnswlib `loadIndex`. |
| `run_cold_loadtime.sh` | `bench_loadtime` with `sync; echo 3 \| sudo tee /proc/sys/vm/drop_caches` before every load. It needs sudo, so run it yourself. |

Typical full-scale commands (M5b), with `B=build-bench/bench` and `D=bench/data/sift`:

```sh
$B/bench_gt verify --base $D/sift_base.fvecs --queries $D/sift_query.fvecs \
  --gt $D/sift_groundtruth.ivecs --json bench/results/gt_verify.json
$B/bench_build --system ours    --data $D/sift_base.fvecs --runs 3 --out ours.hnsw   --json bench/results/build_ours.json
$B/bench_build --system hnswlib --data $D/sift_base.fvecs --runs 3 --out hnswlib.idx --json bench/results/build_hnswlib.json
$B/bench_search --ours ours.hnsw --hnswlib hnswlib.idx --queries $D/sift_query.fvecs \
  --gt $D/sift_groundtruth.ivecs --json bench/results/search.json
$B/bench_loadtime --ours ours.hnsw --hnswlib hnswlib.idx --queries $D/sift_query.fvecs \
  --json bench/results/loadtime_warm.json
bench/run_cold_loadtime.sh build-bench ours.hnsw hnswlib.idx $D/sift_query.fvecs bench/results/loadtime_cold.json
```

For a dev-scale run on a prefix of the base set, pass `--n N` to `bench_build` and use
a ground truth from `bench_gt subset --n N`. `bench_search` refuses a ground truth
whose ids do not exist in the index.

## Method

**Build.** Insertion is single-threaded for both systems, because ours has no
concurrent insert. Both use M=16, efConstruction=200 and seed 100 by default, and the
values used are recorded. One run's time covers index construction through the last
insertion, which includes preallocation for both systems: hnswlib's constructor and
our `reserve()`. Saving is timed separately. VmHWM is read from `/proc/self/status`.
The end value is the peak of the whole process: the dataset plus the largest run.

**Search** (`bench_search`). For each thread count, then each efSearch:

1. Set hnswlib's ef first. `setEf` is not thread-safe, so it is never called while
   searches run.
2. Run one untimed warm-up pass over all queries per system.
3. Run `--reps` timed passes per system (default 5), ordered A B B A A B … across
   repetitions, so that slow drift affects both systems equally.
4. Store every raw pass time, plus the median, min and max. QPS is
   `queries / median pass time`.

Only the query loop is timed. Results go into preallocated arrays. Nothing else
happens inside the timed region: no I/O, no allocation by the harness, no recall
computation. With 1 thread, the pass is a plain loop on the calling thread: ours
calls `search_into` with one `SearchScratch`, and hnswlib calls `searchKnn`. With more
threads, both systems go through the **same** dispatch code: `hnsw::ThreadPool`
`parallel_for` with chunks of 8 queries, a pool created before timing, and timing from
submission to completion. Each pass's results must equal the warm-up pass's
(checked after timing), and a multi-thread run records whether its ids equal the
single-thread ids. hnswlib's max-heap results are converted to ascending order, so
that one recall function serves both systems: |first k found ∩ first k ground-truth
ids| / k, averaged over queries.

**Load time** (`bench_loadtime`). Each strategy is loaded `--reps` times (default 5),
in alternating order. Each measurement times the load, then the first `--first`
queries (default 1000), single-threaded, starting from a fresh scratch. The default
run uses a warm page cache: each strategy is loaded once, untimed, before measuring.
`run_cold_loadtime.sh` drops the cache before every load instead, and the JSON records
which of the two was done. Under WSL2 that drops the Linux VM's page cache only. The
VM's disk image may still be cached by Windows.

## Result files

Every JSON file has `tool`, `environment`, `dataset` and `config` blocks, plus the
tool's measurements. `environment` holds:

- the date and CPU model;
- logical and physical core counts and MemTotal;
- the kernel version, and WSL detection with its evidence;
- the first 20 lines of `lscpu`;
- the compiler used and the g++ and clang++ versions found at configure time;
- the effective common flags, per-file flags and the predefined macros the bench
  code was compiled with (`__OPTIMIZE__`, `NDEBUG`, `__AVX2__`, `__FMA__`), plus a
  `release_build` flag;
- the commit, describe string and dirty state of this repository at build time;
- the hnswlib commit;
- the distance kernel each system uses.

For hnswlib, it records the `USE_SSE` / `USE_AVX` / `USE_AVX512` macros its headers
defined under these flags, its runtime CPU checks, and the actual L2 function its
`L2Space` selected (`avx_path_active`). Tools print a warning when they run from a
non-release or sanitizer build.

`dataset` names the dataset and lists each file used with its size and sha256. A file
recomputed for a subset is marked `in_manifest: false`.

## Fairness notes

Read these next to any comparison:

- **Single-threaded build for both.** hnswlib supports parallel insertion, but we
  don't, so the build-time comparison uses one thread on each side.
- **Index file sizes differ by design.** Our layer-0 records (links plus vector) are
  padded to 64 bytes, so the index file is larger than hnswlib's for the same graph
  parameters. See [docs/FORMAT.md](../docs/FORMAT.md).
- **Per-query memory.** hnswlib allocates a `std::priority_queue` result and
  candidate heaps inside every `searchKnn` call. It also takes a visited list from a
  mutex-protected pool. Ours reuses a caller-owned `SearchScratch` and allocates
  nothing per query once warm. That is part of what is being compared, not a
  harness artifact.
- **Load strategies differ.** Ours can mmap the file and search it in place
  (shallow validation makes load O(1); deep validation reads everything), or read it
  into memory. hnswlib's `loadIndex` always reads the whole file into memory.
- **Distance kernels.** Ours uses an explicit AVX2+FMA kernel chosen at run time.
  hnswlib uses whatever its header detection selects under `-march=native`. That is
  recorded per result file (`environment.hnswlib.l2_function`). In the GCC 15.2 build
  used during development, hnswlib selected `L2SqrSIMD16ExtAVX`. That kernel is written
  with separate multiply and add intrinsics, and in that build GCC fused them into
  `vfmadd` (seen with `objdump -d build-bench/bench/bench_search`), so both sides
  ran 256-bit FMA code there. Recheck after changing the compiler.
- **Same flags.** hnswlib's headers compile inside the bench tools with the
  library's flags (`-O3 -march=native -DNDEBUG` in Release). The bench targets add no
  flags of their own.
- **Thread scratch.** The harness keeps one `SearchScratch` per thread, each on its own
  cache lines. Ours writes its visited-list epoch once per query. hnswlib has no
  equivalent per-thread state to place.
- **VmHWM precision.** On recent kernels VmHWM comes from batched per-CPU RSS
  counters. During development it was seen reading a few hundred kB low after memory
  was freed (`tests/test_bench_util.cpp`). That is negligible next to index sizes,
  but it is not byte-exact.

## Tests

With `HNSW_BUILD_BENCH=ON`, ctest also runs:

- `test_bench_util`: the VmHWM reader, the JSON writer and parser, and the
  median/min/max helper.
- `test_bench_smoke`: the whole pipeline end to end on a 3,000 x 32 synthetic
  dataset. It checks every result file's keys. It also recomputes recall in the test
  by brute force and requires it to equal the harness's value. Finally it requires
  recall ≥ 0.99 at ef=400, multi-thread ids equal to single-thread ids, ascending
  rows of distinct valid ids for both systems, A B B A pass order, and that a
  mismatched ground truth is refused.

In ASan builds, the one translation unit that compiles hnswlib
(`bench/common/hnswlib_index.cpp`) is built without ASan, and UBSan stays on.
hnswlib v0.9.0 reads one id past the end of a neighbor list to compute a prefetch
address (`hnswalg.h`, `searchBaseLayer`, `*(datal + j + 1)`). That is a 4-byte heap
over-read, which ASan reports and aborts on. It is upstream code, so it is exempted
rather than patched. Our own code stays fully instrumented.
