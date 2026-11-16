# hnsw-cpp

From-scratch HNSW approximate nearest neighbor index in C++17.

## Status

Work in progress. See docs/TASK.md.

| Milestone | State |
|---|---|
| M1: CMake project, fvecs/ivecs IO, brute-force kNN, scalar L2, unit tests | done |
| M2: HNSW build and search (scalar, single-threaded baseline) | done |
| M3: contiguous layout, epoch visited list, allocation-free heaps, prefetch, AVX2+FMA kernel | done |
| M4 to M7 | not started |

### Build options (M3)

Each optimization can be switched off for ablation; all default to ON.

| CMake option | OFF behavior |
|---|---|
| `HNSW_OPT_EPOCH_VISITED` | visited marks are a fresh byte array per search pass |
| `HNSW_OPT_ALLOC_FREE_HEAPS` | search heaps are `std::priority_queue` |
| `HNSW_OPT_PREFETCH` | no software prefetch |
| `HNSW_NATIVE` | no `-march=native`; the AVX2 kernel still builds with per-file flags and is chosen at run time |

The distance kernel is chosen at run time: AVX2+FMA when the CPU supports it, else scalar. Pass `hnsw::l2_squared_scalar` explicitly to use the scalar reference.
