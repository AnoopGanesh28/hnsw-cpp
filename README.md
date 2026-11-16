# hnsw-cpp

From-scratch HNSW approximate nearest neighbor index in C++17.

## Status

Work in progress. See docs/TASK.md.

| Milestone | State |
|---|---|
| M1: CMake project, fvecs/ivecs IO, brute-force kNN, scalar L2, unit tests | done |
| M2: HNSW build and search (scalar, single-threaded baseline) | done |
| M3: contiguous layout, epoch visited list, allocation-free heaps, prefetch, AVX2+FMA kernel | done |
| M4: index file format (save, mmap/read load, validation) | in progress: file format done |
| M5 to M7 | not started |

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
