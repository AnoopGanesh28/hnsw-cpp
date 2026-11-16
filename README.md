# hnsw-cpp

From-scratch HNSW approximate nearest neighbor index in C++17.

## Status

Work in progress. See docs/TASK.md.

| Milestone | State |
|---|---|
| M1: CMake project, fvecs/ivecs IO, brute-force kNN, scalar L2, unit tests | done |
| M2: HNSW build and search (scalar, single-threaded baseline) | done |
| M3: contiguous layout, epoch visited list, allocation-free heaps, prefetch, AVX2+FMA kernel | done |
| M4: index file format (save, mmap/read load, validation), thread pool, batch query API | done |
| M5a: benchmark infrastructure (dataset scripts, ground-truth validation, build/search/load-time harness with hnswlib baseline, tests) | done |
| M5b: SIFT1M runs, ablation, profile, plot, results table | partial: false-sharing fix and `bench/run_all.sh` orchestration done; timed runs blocked (`bench/results/env_manual.txt` missing, and timed stages refuse to start without it) |
| M6, M7 | not started |

No benchmark results yet. The tools and the method are described in
[bench/README.md](bench/README.md). SIFT1M downloads and validates on this machine
(`bench/get_sift.sh`), so no synthetic substitute is needed.

### Build options (M3)

Each optimization can be switched off for ablation; all default to ON.

| CMake option | OFF behavior |
|---|---|
| `HNSW_OPT_EPOCH_VISITED` | visited marks are a fresh byte array per search pass |
| `HNSW_OPT_ALLOC_FREE_HEAPS` | search heaps are `std::priority_queue` |
| `HNSW_OPT_PREFETCH` | no software prefetch |
| `HNSW_NATIVE` | no `-march=native`; the AVX2 kernel still builds with per-file flags and is chosen at run time |

The distance kernel is chosen at run time: AVX2+FMA when the CPU supports it, else scalar. Pass `hnsw::l2_squared_scalar` explicitly to use the scalar reference.

### Index files (M4)

`HnswIndex::save(path)` writes the index atomically (temporary file plus rename).
`LoadedIndex::open(path, options)` loads it read-only, either memory-mapped
(`LoadMode::Mmap`, searched in place) or read into memory (`LoadMode::Read`), with the
same search API and bit-identical results. The byte-level format and every check the
loader makes are in [docs/FORMAT.md](docs/FORMAT.md). Files are treated as untrusted:
with `validate_deep` on (the default) any file that loads is safe to search; with it
off, only load files you trust.

### Concurrent search (M4)

`ThreadPool(n_threads)` (`include/hnsw/thread_pool.h`) is a persistent pool reused across
calls; `n_threads = 0` uses `std::thread::hardware_concurrency()`. `search_batch(queries,
n_queries, k, ef, out_ids, out_dists, pool)` on `HnswIndex` and `LoadedIndex` writes
`n_queries * k` results row-major into caller arrays, padding short rows with
`hnsw::kNoNeighbor` (`UINT32_MAX`) and `+infinity`, and returns exactly what sequential
`search` returns. Each pool thread keeps its own `SearchScratch`. Calling `search` from
your own threads at the same time is also safe, as long as each thread uses its own
`SearchScratch` and nothing is inserting.

`-DHNSW_SANITIZE=thread` builds with ThreadSanitizer (not combinable with `address`).
