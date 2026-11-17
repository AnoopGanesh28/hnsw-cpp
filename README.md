# hnsw-cpp

[![CI](https://github.com/AnoopGanesh28/hnsw-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/AnoopGanesh28/hnsw-cpp/actions/workflows/ci.yml)

A from-scratch C++17 implementation of HNSW (Hierarchical Navigable Small World graphs,
Malkov and Yashunin) for approximate nearest-neighbor search under squared L2 distance, on
x86-64. It is built to be measured: it is benchmarked against hnswlib on SIFT1M, and it is
tested under sanitizers and fuzzed, with the evidence saved in the repository.

## Features

- HNSW build and search following the paper's Algorithms 1, 2, 4 and 5 (see [docs/DESIGN.md](docs/DESIGN.md)).
- Contiguous, cache-line-aligned layer-0 records and an index-based graph (no stored pointers).
- Epoch-tagged visited marks, allocation-free heaps, software prefetch: each can be switched off
  at build time for ablation.
- AVX2+FMA distance kernel chosen at run time, with a scalar reference.
- A versioned, checksummed index file format. A file is memory-mapped and searched in place, or read
  into memory, with bit-identical results. Files are treated as untrusted: deep validation makes any
  accepted file safe to search ([docs/FORMAT.md](docs/FORMAT.md)).
- A thread pool and a batch query API; searches on one index may run concurrently.
- Python bindings (pybind11, numpy) with type stubs.
- Benchmarks against hnswlib on SIFT1M with raw results, ablation and a profile.
- ASan, UBSan and TSan runs with GCC and clang, libFuzzer targets for the loader and the vector
  readers, and GitHub Actions CI.

## Status

Milestone state, and the command that reproduces its evidence. Details of every command are in
[Reproducing the results](#reproducing-the-results).

| Milestone | State | Reproduce |
|---|---|---|
| M1: build system, fvecs/ivecs IO, brute-force kNN, scalar L2 | done | `ctest --test-dir build -R "test_io\|test_distance\|test_brute_force\|test_recall"` |
| M2: HNSW build and search (scalar, single-threaded baseline, tag `m2-scalar-baseline`) | done | `ctest --test-dir build -R "test_hnsw_\|test_golden"` |
| M3: contiguous layout, epoch visited list, allocation-free heaps, prefetch, AVX2+FMA kernel | done | `ctest --test-dir build -R "test_layout\|test_scratch\|test_alloc_free\|avx2"` |
| M4: index file format (save, mmap/read load, validation), thread pool, batch API | done | `ctest --test-dir build -R "test_index_file\|test_thread_pool\|test_batch\|test_concurrent"` |
| M5a: benchmark tools, dataset scripts, ground-truth validation, tests | done | `ctest --test-dir build-bench -R bench` (configured with `-DHNSW_BUILD_BENCH=ON`) |
| M5b: SIFT1M runs, ablation, profile, plot, results | done, on one machine (see below) | `bench/run_all.sh`, then `python3 bench/make_report.py --update-readme` and `python3 bench/check_readme.py` |
| M6: Python bindings, tests, example | done | `pip install .` then the Python commands below; `ctest` with `-DHNSW_BUILD_PYTHON=ON` |
| M7a: sanitizers (ASan, UBSan, TSan; GCC and clang), libFuzzer targets, seed corpus, replay tests | done | `scripts/sanitize.sh <gcc\|clang> <mode>`, `scripts/sanitize_python.sh`, `scripts/run_fuzz.sh` |
| M7b: CI, design document, README, quality report | done | `python3 scripts/quality_report.py --check`; CI runs on every push |

What was not measured, or not claimed:

- All benchmark numbers come from one laptop running Linux under WSL2 on Windows, one dataset
  (SIFT1M), one `M` and `efConstruction` pair for both systems, and squared L2 only. The
  cold-cache load times drop the Linux VM's page cache, not the Windows host's, so "cold" means
  cold for Linux (see [Results](#results)).
- The Python bindings have correctness tests but no performance measurement.
- Nothing was measured on non-x86 CPUs, other compilers than GCC and clang, or other operating
  systems; the code is x86-64 only.
- Concurrent insertion does not exist, so build time is single-threaded for both systems.

## Build and test

Requirements (Ubuntu or WSL2 Ubuntu): an x86-64 CPU (AVX2+FMA is used if present), CMake 3.16 or
newer, and GCC or clang with C++17. The project is built and tested in CI with the compilers
that ship in the current `ubuntu-latest` image (GCC 15.2 and clang 21.1 at the time of the saved CI
run). For the Python package, Python 3.10 or newer; for the in-tree Python build, `pybind11` and the
Python development files.

```
sudo apt install build-essential cmake git
git clone https://github.com/AnoopGanesh28/hnsw-cpp.git && cd hnsw-cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure -j
build/example_quickstart            # the C++ example below
```

Tests that need AVX2+FMA exit with the skip code (77) and are reported as skipped on a CPU without it.

The number of ctest entries depends on the options: `HNSW_BUILD_BENCH` adds the benchmark tests and
`HNSW_BUILD_PYTHON` adds the Python tests, except in sanitizer builds. The Quality block below gives the
counts of the canonical configuration and of every sanitizer cell, and explains the difference.

| CMake option | Default | Meaning |
|---|---|---|
| `HNSW_NATIVE` | ON | compile with `-march=native`. OFF builds a portable binary; the AVX2 kernel still builds with per-file flags and is chosen at run time |
| `HNSW_OPT_EPOCH_VISITED` | ON | OFF: visited marks are a fresh byte array per search pass |
| `HNSW_OPT_ALLOC_FREE_HEAPS` | ON | OFF: search heaps are `std::priority_queue` |
| `HNSW_OPT_PREFETCH` | ON | OFF: no software prefetch |
| `HNSW_SANITIZE` | off | `address`, `undefined`, `address,undefined` or `thread` (thread cannot be combined with address) |
| `HNSW_BUILD_BENCH` | OFF | build the benchmark tools (needs `bench/get_hnswlib.sh` first) |
| `HNSW_BUILD_PYTHON` | OFF | build the Python extension and register its tests (needs pybind11 and Python development files) |
| `HNSW_BUILD_FUZZ` | OFF | build the libFuzzer targets (clang with `HNSW_SANITIZE=address,undefined`) |

Release builds use `-O3`; warnings are on (`-Wall -Wextra`).

## Usage

### C++

[examples/quickstart.cpp](examples/quickstart.cpp), built as `example_quickstart`:

```cpp
hnsw::HnswParams params;
params.M = 16;
params.ef_construction = 200;
hnsw::HnswIndex index(dim, params);                    // squared L2, fastest kernel for this CPU
for (size_t i = 0; i < n; ++i) index.add(base + i * dim);

hnsw::KnnResult r = index.search(query, k, ef);        // r.ids, r.dists (squared), ascending
hnsw::ThreadPool pool(0);                              // 0 = one thread per hardware thread
index.search_batch(queries, nq, k, ef, ids, dists, pool);

index.save("index.hnsw");                              // atomic: temporary file, then rename
hnsw::LoadedIndex loaded = hnsw::LoadedIndex::open("index.hnsw");   // mmap, deep validation
loaded.search_batch(queries, nq, k, ef, ids, dists, pool);          // bit-identical results
```

`LoadOptions` selects `LoadMode::Read` instead of mmap, or `validate_deep = false` for files you
trust. `LoadedIndex::from_memory(data, size)` validates and searches a file image already in memory.

### Python

```
python3 -m venv .venv && . .venv/bin/activate
pip install .                      # scikit-build-core builds a portable (no -march=native) wheel
python python/examples/quickstart.py
python -m unittest discover -s python/tests
```

```python
import numpy as np, hnsw_cpp

base = np.random.default_rng(0).random((5000, 64), dtype=np.float32)
index = hnsw_cpp.Index(64, M=16, ef_construction=200)
index.add(base)
ids, sq_dists = index.search(base[:10], 10, 100)         # squared L2; missing results are id -1
index.save("index.hnsw")
with hnsw_cpp.load("index.hnsw", mode="mmap", validate="deep") as loaded:
    ids2, _ = loaded.search(base[:10], 10, 100, threads=0)
```

`Index` offers build, `add`, `search` and `save`; `load(path, mode, validate)` returns a read-only
`LoadedIndex`. Inputs must already be C-contiguous float32 numpy arrays; nothing is copied or converted,
and wrong input raises `TypeError` or `ValueError` saying how to fix it. Any number of searches may run
at once; `add` is exclusive. `validate="shallow"` requires a trusted file.

### Index files

`HnswIndex::save` writes the in-memory arena verbatim, so a mapped file is searched in place with no
translation. The byte-level format and every check the loader makes are in
[docs/FORMAT.md](docs/FORMAT.md). The design and its trade-offs are in [docs/DESIGN.md](docs/DESIGN.md).

## Results

Measured on SIFT1M against hnswlib on one laptop. Every number below is generated by
`bench/make_report.py` from the raw files in [bench/results/](bench/results/), which
`bench/run_all.sh` wrote. `bench/check_readme.py` checks that each of those numbers
appears in `bench/results/summary.md`, and that the text outside the generated block
contains no numbers of its own. The method is described in [bench/README.md](bench/README.md).

<!-- RESULTS:BEGIN -->
<!-- Generated by bench/make_report.py --update-readme; do not edit. -->

Generated from [bench/results/summary.md](bench/results/summary.md), which also lists the full environment. Values are truncated, not rounded, to the digits shown.

**Verdict** (faster or slower only where the pass-QPS ranges do not overlap): Single-threaded, ours is faster than hnswlib at recall@10 >= 0.95 (7436 vs 6158 median QPS, ratio 1.207); ours is faster than hnswlib at recall@10 >= 0.99 (3254 vs 2665 median QPS, ratio 1.220). With 4 threads, ours is faster than hnswlib at recall@10 >= 0.95 (14990 vs 13469 median QPS, ratio 1.112); ours is faster than hnswlib at recall@10 >= 0.99 (6029 vs 5496 median QPS, ratio 1.096). With 8 threads, the difference between ours and hnswlib is not distinguishable from noise (ranges overlap) at recall@10 >= 0.95 (18475 vs 17865 median QPS, ratio 1.034); the difference between ours and hnswlib is not distinguishable from noise (ranges overlap) at recall@10 >= 0.99 (7383 vs 7215 median QPS, ratio 1.023).

**Noise.** Repeating the 4 headline single-thread measurements at the end of the run gave repeat / sweep median-QPS ratios from 0.994 to 1.071. 34 measured points have a pass spread above 15% ((max - min) / median): 10 in `sweep_t1.json`, 6 in `sweep_t4.json`, 12 in `sweep_t8.json`, 2 in `repeat_t1.json`, 4 in `ablation`; 21 of the 28 flagged sweep points are at ef <= 64, where a pass is shortest. They are listed under High-spread points.

### Headline: QPS at the first ef reaching each recall target

Each system at its own first swept ef whose recall@10 reaches the target (no interpolation).

| threads | recall@10 target | system | ef | recall@10 | QPS (median) | QPS range (slowest to fastest pass) |
|---|---|---|---|---|---|---|
| 1 | >= 0.95 | ours | 56 | 0.9544 | 7436 | 6802 to 8083 |
| 1 | >= 0.95 | hnswlib | 56 | 0.9545 | 6158 | 5634 to 6644 |
| 1 | >= 0.99 | ours | 160 | 0.9931 | 3254 | 3002 to 3299 |
| 1 | >= 0.99 | hnswlib | 160 | 0.9932 | 2665 | 2459 to 2720 |
| 4 | >= 0.95 | ours | 56 | 0.9544 | 14990 | 14035 to 15118 |
| 4 | >= 0.95 | hnswlib | 56 | 0.9545 | 13469 | 13228 to 13690 |
| 4 | >= 0.99 | ours | 160 | 0.9931 | 6029 | 5870 to 6068 |
| 4 | >= 0.99 | hnswlib | 160 | 0.9932 | 5496 | 4948 to 5598 |
| 8 | >= 0.95 | ours | 56 | 0.9544 | 18475 | 17615 to 18851 |
| 8 | >= 0.95 | hnswlib | 56 | 0.9545 | 17865 | 15569 to 18013 |
| 8 | >= 0.99 | ours | 160 | 0.9931 | 7383 | 7204 to 7660 |
| 8 | >= 0.99 | hnswlib | 160 | 0.9932 | 7215 | 6909 to 7353 |

A system is called faster or slower only when the two systems' pass-QPS ranges (slowest to fastest pass) do not overlap; otherwise the comparison is reported as not distinguishable from noise. The median ratio is shown either way.

| threads | recall@10 target | ours QPS / hnswlib QPS (median) | ours is |
|---|---|---|---|
| 1 | >= 0.95 | 1.207 | faster |
| 1 | >= 0.99 | 1.220 | faster |
| 4 | >= 0.95 | 1.112 | faster |
| 4 | >= 0.99 | 1.096 | faster |
| 8 | >= 0.95 | 1.034 | not distinguishable from noise (ranges overlap) |
| 8 | >= 0.99 | 1.023 | not distinguishable from noise (ranges overlap) |

Multi-thread result ids identical to single-thread (same system and ef, compared by hash):

| threads | identical points |
|---|---|
| 4 | 50 of 50 |
| 8 | 50 of 50 |

![recall vs QPS](bench/results/recall_vs_qps.png)

### Build, index size, memory

| system | vectors | M | efConstruction | build s (each run, in order) | build s (median) | build s (range) | index bytes on disk | index MiB |
|---|---|---|---|---|---|---|---|---|
| ours | 1000000 | 16 | 200 | 425.2, 420.7, 409.4 | 420.7 | 409.4 to 425.2 | 717532052 | 684.2 |
| hnswlib | 1000000 | 16 | 200 | 438.6, 441.6, 437.5 | 438.6 | 437.5 to 441.6 | 660565344 | 629.9 |

Peak memory of the build process (VmHWM, kB); the dataset-only value is VmHWM right after loading the base vectors, and the delta is the peak above it:

| system | VmHWM end | VmHWM after dataset load (dataset only) | delta |
|---|---|---|---|
| ours | 1230752 | 504568 | 726184 |
| hnswlib | 1244412 | 504548 | 739864 |

### Load time

**Warm page cache** (each strategy loaded once untimed first), 5 repetitions; first 1000 queries at ef 100, single-threaded, fresh scratch:

| strategy | load ms (median) | load ms (range) | first 1000 queries ms (median) | first 1000 queries ms (range) |
|---|---|---|---|---|
| ours-mmap-shallow | 0.0324 | 0.0319 to 0.0336 | 205 | 204 to 210 |
| ours-mmap-deep | 76.4 | 75.2 to 84.8 | 208 | 206 to 222 |
| ours-read-shallow | 251 | 245 to 281 | 258 | 253 to 268 |
| ours-read-deep | 327 | 326 to 405 | 267 | 252 to 286 |
| hnswlib-load | 671 | 658 to 867 | 251 | 248 to 262 |

**Cold page cache**: `sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null` before every load, 5 repetitions, same query settings:

| strategy | load ms (median) | load ms (range) | first 1000 queries ms (median) | first 1000 queries ms (range) |
|---|---|---|---|---|
| ours-mmap-shallow | 9.39 | 6.16 to 12.3 | 1332 | 1058 to 1495 |
| ours-mmap-deep | 466 | 455 to 1141 | 312 | 284 to 352 |
| ours-read-shallow | 497 | 467 to 845 | 352 | 313 to 433 |
| ours-read-deep | 656 | 543 to 845 | 372 | 356 to 442 |
| hnswlib-load | 1108 | 879 to 1165 | 430 | 273 to 573 |

**Cold vs warm**, time until the first 1000 queries are answered (median load + median first-queries time):

| strategy | warm ms | cold ms |
|---|---|---|
| ours-mmap-shallow | 205 | 1342 |
| ours-mmap-deep | 284 | 779 |
| ours-read-shallow | 510 | 849 |
| ours-read-deep | 594 | 1029 |
| hnswlib-load | 922 | 1538 |

WSL2 caveat: dropping caches empties the Linux VM's page cache only. The VM's virtual disk is a file on the Windows host, which may still hold it in the Windows file cache, so cold here means cold for Linux, not necessarily a read from the physical drive. With mmap, the cold cost moves from the load into the first queries, which fault pages in on demand.

### Ablation

Single-threaded, pinned, the same `ours.hnsw` for every variant; 5 repetitions per combination, each its own process (1 warm-up + 1 timed pass), combinations interleaved round robin. QPS is queries / median timed pass. Every ratio is of medians; it is marked "within noise" when the QPS ranges (slowest to fastest repetition) of the two configurations overlap, and such a ratio is not a measured speedup or slowdown. Recall@10 per kernel: ef 56: scalar 0.9544, AVX2 0.9544; ef 160: scalar 0.9931, AVX2 0.9931.

**Each optimization alone** (QPS; ratio over all off with the scalar kernel):

| optimization | ef 56 (recall >= 0.95) | ef 160 (recall >= 0.99) |
|---|---|---|
| epoch visited list | 3916 (x1.068) | 1641 (x1.004, within noise) |
| allocation-free heaps | 3744 (x1.021, within noise) | 1589 (x0.972, within noise) |
| software prefetch | 4924 (x1.343) | 2155 (x1.318) |
| AVX2+FMA kernel | 6289 (x1.716) | 2736 (x1.674) |

Baseline, all off (scalar): ef 56: 3665 QPS (3477 to 3750), ef 160: 1634 QPS (1611 to 1644).

**Cumulative** (QPS; total ratio over all off, and the step over the row above):

| configuration | ef 56 (recall >= 0.95) | ef 160 (recall >= 0.99) |
|---|---|---|
| all off (scalar) | 3665 | 1634 |
| + epoch visited list | 3916 (total x1.068; step x1.068) | 1641 (total x1.004, within noise; step x1.004, within noise) |
| + allocation-free heaps | 4026 (total x1.098, within noise; step x1.028, within noise) | 1672 (total x1.023, within noise; step x1.018, within noise) |
| + software prefetch | 5336 (total x1.456; step x1.325) | 2234 (total x1.367; step x1.336) |
| + AVX2+FMA kernel | 7839 (total x2.138; step x1.468) | 3172 (total x1.940; step x1.419) |

**Leave one out** from the full configuration (ef 56: 7839 QPS (7717 to 7962), ef 160: 3172 QPS (3104 to 3249)):

| configuration | ef 56 (recall >= 0.95) | ef 160 (recall >= 0.99) |
|---|---|---|
| without epoch visited list | 7221 (full / this: x1.085) | 3185 (full / this: x0.995, within noise) |
| without allocation-free heaps | 7783 (full / this: x1.007, within noise) | 3119 (full / this: x1.016, within noise) |
| without software prefetch | 6968 (full / this: x1.124) | 2835 (full / this: x1.118) |
| without AVX2+FMA kernel | 5336 (full / this: x1.468) | 2234 (full / this: x1.419) |

All 16 combinations (QPS median, slowest to fastest repetition):

| enabled | ef 56 (recall >= 0.95) | ef 160 (recall >= 0.99) |
|---|---|---|
| all off (scalar) | 3665 (3477 to 3750) | 1634 (1611 to 1644) |
| simd | 6289 (5809 to 6370) | 2736 (2391 to 2782) |
| prefetch | 4924 (3930 to 4966) | 2155 (1962 to 2192) |
| prefetch+simd | 7058 (6625 to 7135) | 3056 (2941 to 3107) |
| heaps | 3744 (3652 to 3793) | 1589 (1492 to 1658) |
| heaps+simd | 6410 (5542 to 6436) | 2599 (2499 to 2839) |
| heaps+prefetch | 4724 (4420 to 4976) | 2181 (1964 to 2194) |
| heaps+prefetch+simd | 7221 (7175 to 7298) | 3185 (3053 to 3204) |
| epoch | 3916 (3868 to 3946) | 1641 (1514 to 1649) |
| epoch+simd | 6599 (6543 to 6916) | 2767 (2732 to 2795) |
| epoch+prefetch | 5333 (4813 to 5349) | 2208 (1971 to 2230) |
| epoch+prefetch+simd | 7783 (7150 to 7814) | 3119 (3057 to 3155) |
| epoch+heaps | 4026 (3559 to 4032) | 1672 (1489 to 1689) |
| epoch+heaps+simd | 6968 (6023 to 6999) | 2835 (2740 to 2864) |
| epoch+heaps+prefetch | 5336 (5252 to 5431) | 2234 (1967 to 2264) |
| epoch+heaps+prefetch+simd | 7839 (7717 to 7962) | 3172 (3104 to 3249) |

Result-id check (`ablation_check.json`): identical ids for every same-kernel combination: True.

![ablation](bench/results/ablation.png)

### Layout ablation

Memory layout, measured on the full base (1000000 vectors). The M2 build time was projected from a 100000-vector build (51.4 s) to 617.0 s for 1000000 vectors (n log n; limit 5400.0 s). M2 baseline (pre-layout, scalar kernel) versus the all-off scalar variant (current layout) on an index with the same graph, both single-threaded and pinned.

How the samples were taken: one M2 process built the index once, then ran 7 repetitions M_0 .. M_6 (each 1 warm-up + 1 timed pass per ef). Before M_0 and after every M_r it ran the all-off variant once as a separate process, giving 8 all-off samples A_0 .. A_7 in the order A_0 M_0 A_1 M_1 ... M_6 A_7. M_r sits between A_r and A_(r+1). The "before" group A_0 .. A_6 and the "after" group A_1 .. A_7 share 6 of their 7 samples (A_1 .. A_6), so they are not independent: they differ only in A_0 versus A_7. At ef 56 the medians are A_4 (2.6936 s) and A_1 (2.6933 s), close enough that both truncate to the same QPS. At ef 160 both groups' median is the same sample, A_1, so the two values are identical. The comparison below therefore uses all 8 all-off samples as one group, and lists both overlapping groups and every raw sample.

| ef | recall@10 | M2 QPS median (range) | all-off QPS median (range), all 8 samples | all-off / M2 (medians) | current layout is |
|---|---|---|---|---|---|
| 56 | 0.9544 | 2933 (2907 to 2942) | 3712 (3381 to 3755) | x1.265 | faster |
| 160 | 0.9931 | 1214 (1093 to 1262) | 1635 (1550 to 1648) | x1.346 | faster |

Overlapping groups (not independent; shown only to explain the identical values reported before):

| ef | before A_0..A_6 QPS | after A_1..A_7 QPS | before median s | after median s |
|---|---|---|---|---|
| 56 | 3712 | 3712 | 2.6936 | 2.6933 |
| 160 | 1638 | 1638 | 6.1032 | 6.1032 |

Every raw timed pass, in execution order (seconds for 10000 queries):

| run | ef 56 s | ef 160 s |
|---|---|---|
| A_0 | 2.7521 | 6.4514 |
| M_0 | 3.4089 | 7.9668 |
| A_1 | 2.6933 | 6.1032 |
| M_1 | 3.4077 | 8.2311 |
| A_2 | 2.6892 | 6.1683 |
| M_2 | 3.4005 | 9.1462 |
| A_3 | 2.7086 | 6.1378 |
| M_3 | 3.3980 | 7.9353 |
| A_4 | 2.6936 | 6.0651 |
| M_4 | 3.4291 | 8.9694 |
| A_5 | 2.9570 | 6.0970 |
| M_5 | 3.4398 | 7.9234 |
| A_6 | 2.6628 | 6.0807 |
| M_6 | 3.4129 | 9.0236 |
| A_7 | 2.6829 | 6.1251 |

Result-id check (`layout_check.json`): M2 ids equal the all-off variant's: True.

### Profile

Sampled with `perf record -e cpu-clock` (user-space samples, whole process: includes loading and the untimed warm-up and recall code), single-threaded and pinned. Top symbols (`perf report --stdio --no-children`, full lists in `bench/results/perf/report_*.txt`):

ours:

| samples | object | symbol |
|---|---|---|
| 56.41% | bench_search | `hnsw::l2_squared_avx2` |
| 41.90% | bench_search | `hnsw::detail::search_layer` |
| 1.24% | bench_search | `hnsw::detail::greedy_closest` |
| 0.06% | bench_search | `hnsw::search_graph_into` |
| 0.02% | bench_search | `double ::timed_pass [clone .isra.0]` |
| 0.02% | bench_search | `std::vector::reserve` |
| 0.01% | bench_search | `hnsw::LoadedIndex::search_into const` |
| 0.01% | sha256sum | `0x0000000000301bb7` |

hnswlib:

| samples | object | symbol |
|---|---|---|
| 51.18% | bench_search | `hnswlib::HierarchicalNSW::searchKnn const` |
| 46.55% | bench_search | `hnswlib::L2SqrSIMD16ExtAVX` |
| 0.35% | bench_search | `hnswlib::HierarchicalNSW::loadIndex` |
| 0.26% | libc.so.6 | `llseek@GLIBC_2.2.5` |
| 0.14% | libc.so.6 | `malloc` |
| 0.11% | bench_search | `std::_Hashtable::_M_insert_unique_node` |
| 0.10% | libc.so.6 | `cfree@GLIBC_2.2.5` |
| 0.09% | libstdc++.so.6.0.35 | `std::basic_filebuf::seekoff` |

`perf stat` over the same runs (user space):

| system | task-clock ms | cycles | instructions | IPC | cache misses / references |
|---|---|---|---|---|---|
| ours | 13861 | 57530379557 | 24600158222 | 0.42 | 81.2% |
| hnswlib | 17862 | 71999932708 | 29624829963 | 0.41 | 78.5% |

Hardware counters: exposed (user space).

```
perf: /usr/bin/perf
perf version 7.0.14
binary: build-prof/bench/bench_search
pinned cpu: 2
ef: 56 (smallest ef with ours recall@10 >= 0.95 in sweep_t1.json)
perf_event_paranoid: 2
```

### Reproducibility

Headline single-thread points measured again at the end of the run (`repeat_t1.json`) against the main sweep (`sweep_t1.json`):

| system | ef | sweep QPS | repeat QPS | repeat / sweep | same result ids |
|---|---|---|---|---|---|
| hnswlib | 56 | 6158 | 6568 | 1.066 | True |
| hnswlib | 160 | 2665 | 2664 | 0.999 | True |
| ours | 56 | 7436 | 7964 | 1.071 | True |
| ours | 160 | 3254 | 3237 | 0.994 | True |

### High-spread points

Points whose timed passes spread by more than 15% ((max - min) / median):

| file | point | passes | spread |
|---|---|---|---|
| sweep_t1.json | ours ef 12 threads 1 | 7 | 32.5% |
| sweep_t1.json | hnswlib ef 12 threads 1 | 7 | 42.3% |
| sweep_t1.json | ours ef 14 threads 1 | 7 | 26.1% |
| sweep_t1.json | hnswlib ef 20 threads 1 | 7 | 20.8% |
| sweep_t1.json | ours ef 24 threads 1 | 7 | 19.1% |
| sweep_t1.json | hnswlib ef 24 threads 1 | 7 | 22.3% |
| sweep_t1.json | ours ef 40 threads 1 | 7 | 37.2% |
| sweep_t1.json | ours ef 56 threads 1 | 7 | 17.3% |
| sweep_t1.json | hnswlib ef 56 threads 1 | 7 | 16.6% |
| sweep_t1.json | hnswlib ef 96 threads 1 | 7 | 17.4% |
| sweep_t4.json | hnswlib ef 10 threads 4 | 7 | 17.7% |
| sweep_t4.json | ours ef 12 threads 4 | 7 | 23.0% |
| sweep_t4.json | hnswlib ef 12 threads 4 | 7 | 40.2% |
| sweep_t4.json | ours ef 16 threads 4 | 7 | 15.9% |
| sweep_t4.json | hnswlib ef 20 threads 4 | 7 | 16.5% |
| sweep_t4.json | ours ef 64 threads 4 | 7 | 16.4% |
| sweep_t8.json | ours ef 10 threads 8 | 7 | 24.3% |
| sweep_t8.json | hnswlib ef 10 threads 8 | 7 | 44.6% |
| sweep_t8.json | ours ef 16 threads 8 | 7 | 19.7% |
| sweep_t8.json | hnswlib ef 28 threads 8 | 7 | 16.8% |
| sweep_t8.json | hnswlib ef 32 threads 8 | 7 | 17.1% |
| sweep_t8.json | hnswlib ef 56 threads 8 | 7 | 15.5% |
| sweep_t8.json | hnswlib ef 128 threads 8 | 7 | 32.9% |
| sweep_t8.json | hnswlib ef 200 threads 8 | 7 | 15.7% |
| sweep_t8.json | ours ef 320 threads 8 | 7 | 16.6% |
| sweep_t8.json | hnswlib ef 320 threads 8 | 7 | 17.3% |
| sweep_t8.json | ours ef 400 threads 8 | 7 | 18.2% |
| sweep_t8.json | hnswlib ef 640 threads 8 | 7 | 17.9% |
| repeat_t1.json | ours ef 56 threads 1 | 7 | 33.3% |
| repeat_t1.json | hnswlib ef 56 threads 1 | 7 | 24.5% |
| ablation | simd ef 160 | 5 | 16.0% |
| ablation | prefetch ef 56 | 5 | 26.1% |
| ablation | heaps+simd ef 56 | 5 | 16.0% |
| ablation | epoch+heaps+simd ef 56 | 5 | 16.1% |

<!-- RESULTS:END -->

### How to read these numbers

- **Machine.** A laptop running Linux under WSL2 on Windows. Background Windows
  activity, power management and the WSL2 virtual machine all add noise that a
  dedicated Linux server would not have. Single-thread runs are pinned to one logical
  CPU, and multi-thread runs are not pinned. The power and session settings recorded
  by hand are listed in `bench/results/summary.md`.
- **Noise.** Each point is the median of several timed passes, and the spread of every
  point is stored. The generated "Noise" paragraph above states the measured run-to-run
  variation and where the high-spread points are. A comparison is called faster or
  slower only when the pass ranges of the two sides do not overlap; otherwise the
  generated text says it is not distinguishable from noise.
- **Cold-cache load time.** Measured with `bench/run_cold_loadtime.sh`, which drops the
  page cache (with `sudo`) before every load. Under WSL2 that clears the Linux VM's page
  cache only; Windows may still have the virtual disk cached, so "cold" means cold for
  Linux.

### Fairness notes

- Both systems build single-threaded, because ours has no concurrent insert. hnswlib
  can build in parallel.
- Index file sizes differ by design: our layer-zero records are padded to a cache line.
- hnswlib allocates its result and candidate heaps inside every query and takes a
  visited list from a mutex-protected pool. Ours reuses a caller-owned scratch and
  allocates nothing per query once warm. That difference is part of what is compared.
- Ours can map the index file and search it in place, or read it into memory. hnswlib's
  `loadIndex` always reads the whole file.
- Ours uses an explicit AVX2+FMA kernel. hnswlib uses the kernel its headers select
  under `-march=native`; the result files record which one (`L2SqrSIMD16ExtAVX` here).
- Both compile with the same flags in the same binaries. Multi-thread runs use the same
  thread pool and dispatch code for both systems.
- Peak memory comes from the kernel's VmHWM, which is not byte-exact.

## Quality

Everything in this block is generated by `scripts/quality_report.py` from the files in
[docs/evidence/](docs/evidence/) (raw ctest, sanitizer, Python-under-ASan, fuzz and CI outputs).
`python3 scripts/quality_report.py --check` fails if the block disagrees with those files, and CI runs it.
The CI record is a run of the commit it names; it is refreshed from `gh run view` and cannot describe the
commit that contains it.

<!-- QUALITY:BEGIN -->
_Generated by `scripts/quality_report.py` from `docs/evidence/`; do not edit by hand. `scripts/quality_report.py --check` fails if this block disagrees with those files._

#### Tests (ctest)

| Configuration | Build | Options | ctest entries | Passed | Failed | Skipped |
|---|---|---|---:|---:|---:|---:|
| gcc, canonical full | Release -O3 | BENCH=ON, PYTHON=ON, FUZZ=OFF, SANITIZE=off | 27 | 27 | 0 | 0 |
| clang, canonical full | Release -O3 | BENCH=ON, PYTHON=ON, FUZZ=OFF, SANITIZE=off | 27 | 27 | 0 | 0 |
| gcc, address | -O1 -g | SANITIZE=address, BENCH=ON, PYTHON=ON | 25 | 25 | 0 | 0 |
| gcc, undefined | -O1 -g | SANITIZE=undefined, BENCH=ON, PYTHON=ON | 25 | 25 | 0 | 0 |
| gcc, address+undefined | -O1 -g | SANITIZE=address,undefined, BENCH=ON, PYTHON=ON | 25 | 25 | 0 | 0 |
| gcc, thread | -O1 -g | SANITIZE=thread, BENCH=ON, PYTHON=ON | 25 | 25 | 0 | 0 |
| clang, address | -O1 -g | SANITIZE=address, BENCH=ON, PYTHON=ON | 25 | 25 | 0 | 0 |
| clang, undefined | -O1 -g | SANITIZE=undefined, BENCH=ON, PYTHON=ON | 25 | 25 | 0 | 0 |
| clang, address+undefined | -O1 -g | SANITIZE=address,undefined, BENCH=ON, PYTHON=ON | 25 | 25 | 0 | 0 |
| clang, thread | -O1 -g | SANITIZE=thread, BENCH=ON, PYTHON=ON | 25 | 25 | 0 | 0 |

The canonical runs have 27 ctest entries and every sanitizer cell 25. The 25 entries in the sanitizer cells are the same entries as in the canonical runs; the 2 that exist only in the canonical runs are `python_quickstart`, `python_tests`. The sanitizer cells were configured with HNSW_BUILD_PYTHON=ON, so the extension was built there, but its tests are not registered with ctest in a sanitizer build (the interpreter is not instrumented). They are run separately under ASan, below.

| Canonical run | Checks (C++ test assertions) | Failed | Entries reporting a count | Python unittests |
|---|---:|---:|---:|---|
| gcc | 21,227 | 0 | 23 of 27 | 33 ran, OK |
| clang | 21,227 | 0 | 23 of 27 | 33 ran, OK |

Checks are the `N checks, M failed` lines the C++ tests print; entries that do not use that helper (the Python tests, the quickstart, the replay tests, the benchmark tests) report no count and are not in the sum.

#### Sanitizer matrix

Full ctest suite per cell (passed/total); UBSan aborts on any finding, ASan runs with leak detection.

| Compiler | address | undefined | address+undefined | thread |
|---|---:|---:|---:|---:|
| gcc | 25/25 | 25/25 | 25/25 | 25/25 |
| clang | 25/25 | 25/25 | 25/25 | 25/25 |

Python tests on the ASan-built extension (interpreter not instrumented, runtime preloaded):

| Compiler | unittests | Result |
|---|---:|---|
| gcc | 33 | OK |
| clang | 33 | OK |

#### Fuzzing (libFuzzer, clang, ASan+UBSan)

| Target | Wall time | Workers | Executions | Crashes | OOMs | Timeouts | Edges | Features | Committed corpus |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fuzz_loader | 1807 s | 4 | 9,462,526 | 0 | 0 | 0 | 599 | 1,716 | 229 files, 1,010,553 bytes |
| fuzz_fvecs | 301 s | 4 | 7,777,855 | 0 | 0 | 0 | 162 | 359 | 23 files, 3,039 bytes |

Loader edge coverage plateaued early: 99% of the final 599 edges (at least 594) was reached by 4 s, and the final value was first seen at 1132 s of 1807 s. Features kept growing: 1,600 at the 56 s sample, 1,716 at the end. Read from the statistics lines of the log.

Effectiveness check: with this line removed from deep validation (`if (x >= n) {`), fuzz_loader started from the seeds alone stopped on AddressSanitizer `SEGV` at a recorded wall time of 0 s (whole seconds; limit 300 s). The code was restored and verified with cmp. This was an easy bug to find, so it shows the setup works, not that it finds subtle ones.

#### Continuous integration

Workflow `CI`, run [44112293359](https://github.com/AnoopGanesh28/hnsw-cpp/actions/runs/44112293359) on commit `cade0f2` ("ci: run the C++ example and check the README against its evidence"), event `push`: **success**. Later commits are not covered by this record.

| Job | Conclusion |
|---|---|
| README matches its evidence | success |
| address+undefined (clang) | success |
| bench-smoke | success |
| fuzz-smoke | success |
| python 3.10 | success |
| python 3.12 | success |
| python 3.13 | success |
| python under ASan | success |
| release (clang) | success |
| release (gcc) | success |
| thread (clang) | success |
| thread (gcc) | success |
<!-- QUALITY:END -->

## Reproducing the results

Every figure in this README comes from a file in `bench/results/` or `docs/evidence/` that one of the
commands below wrote. Replace `build-*` directory names as you like.

**Tests, canonical full configuration** (Release, benchmarks and Python on; needs hnswlib and pybind11):

```
bench/get_hnswlib.sh                          # hnswlib v0.9.0 at a pinned commit, into bench/third_party/
scripts/ctest_full.sh gcc                     # writes docs/evidence/ctest_gcc_release.txt
scripts/ctest_full.sh clang
```

**Sanitizers** (each builds a fresh tree and runs the full suite; the Python run needs the ASan tree from the first):

```
scripts/sanitize.sh gcc address               # also: undefined, address+undefined, thread; and clang
scripts/sanitize_python.sh gcc                # Python tests on the ASan-built extension
```

**Fuzzing** (clang with libFuzzer; install `llvm` for symbolized traces):

```
cmake -S . -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ -DHNSW_SANITIZE=address,undefined -DHNSW_BUILD_FUZZ=ON
cmake --build build-fuzz -j
scripts/run_fuzz.sh build-fuzz loader 1800 4  # log: docs/evidence/fuzz_loader.log; minimizes the corpus
scripts/run_fuzz.sh build-fuzz fvecs 300 4
python3 scripts/fuzz_summary.py docs/evidence/fuzz_loader.log docs/evidence/fuzz_fvecs.log \
  > docs/evidence/fuzz_summary.json
```

Seeds, the minimized corpus and past crashes are replayed by `ctest` in every build
(`fuzz_replay_loader`, `fuzz_replay_fvecs`); see [tests/fuzz/README.md](tests/fuzz/README.md).

**CI record and the Quality block:**

```
gh run view <run-id> --json databaseId,workflowName,displayTitle,event,headBranch,headSha,status,conclusion,createdAt,updatedAt,url,jobs \
  > docs/evidence/ci_latest.json
python3 scripts/quality_report.py --update-readme
```

**Benchmarks** (long-running; `bench/results/stage_times.tsv` records how long each stage took on the
machine that produced the committed results; do not run other heavy work meanwhile). Needs `python3` with
`matplotlib`, `taskset`, and `perf` for the profile stage. Timed stages refuse to start without
`bench/results/env_manual.txt`, a hand-written note of the machine's power and session settings, which
every result file embeds.

```
bench/get_hnswlib.sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DHNSW_BUILD_BENCH=ON
cmake --build build-bench -j
bench/get_sift.sh build-bench                 # downloads and validates SIFT1M into bench/data/sift/
bench/run_all.sh                              # stages: build s0 s1 s2 s3 s4 s5 s6 s7 (see its header)
bench/run_all.sh s3                           # prints the cold-cache command (needs sudo): run it yourself,
                                              # it is also saved in bench/results/cold_loadtime_command.txt
python3 bench/make_report.py --update-readme  # summary.md, plots and the Results block
python3 bench/check_readme.py                 # every number in the block is traceable; no numbers elsewhere
```

The methods and every tool's flags are in [bench/README.md](bench/README.md). The golden test's
expected values are regenerated by `tests/gen_golden.sh` (it builds the `m2-scalar-baseline` tag).

## Limitations

- x86-64 only; the file format is little-endian and big-endian hosts are refused.
- Squared L2 only. The format has a metric field but defines only one metric.
- Insertion is single-threaded; there are no deletes, no updates and no filtered search.
- The graph must fit in memory (or be memory-mapped from a file that does): there is no out-of-core
  construction.
- Index files are larger than hnswlib's because records are padded to cache-line boundaries.
- A memory-mapped index file must not be truncated or modified while it is loaded.
- With many exact duplicate vectors, search can return fewer than `k` results (a known HNSW behavior,
  covered by a test); see [docs/DESIGN.md](docs/DESIGN.md).
- In Python, the shared lock favors readers, so `add` can wait under a continuous stream of searches; the
  type stub is hand-written.
- Benchmark results are from one machine under WSL2 and are noisy; the Results section says which
  comparisons are distinguishable from noise and which are not.

## Future work

- Concurrent inserts.
- Deletes and updates.
- Filtered search.
- GPU search or construction.
- Graphs larger than RAM beyond what mmap gives.
- Quantization (scalar or product) for smaller indexes.
- Inner-product and cosine metrics.

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md): how the implementation follows the paper, the memory layout, search path,
  concurrency, validation and fuzzing, testing, benchmark method, known limits.
- [docs/FORMAT.md](docs/FORMAT.md): the index file format and everything the loader checks.
- [bench/README.md](bench/README.md): benchmark tools and method.
- [tests/fuzz/README.md](tests/fuzz/README.md): fuzz targets, corpus and how to run them.
- [docs/TASK.md](docs/TASK.md): the original project specification.
