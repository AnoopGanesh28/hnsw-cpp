# Task: Build an HNSW approximate-nearest-neighbor index in C++17, end to end, in this session

Build a from-scratch HNSW index (Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs") with AVX2 SIMD distance kernels, an mmap'd on-disk index, and multithreaded search, then benchmark it honestly against hnswlib on SIFT1M. Work autonomously through the milestones in order. Don't stop to ask questions unless something is blocked or a decision cannot be reversed.

Target: Linux, x86-64 with AVX2 + FMA, GCC/Clang, CMake.

## Working rules
1. Never invent, estimate, or round up a number. Every figure in the README comes from a script in /bench whose raw output is saved in /bench/results.
2. Each milestone ends with passing tests and its own commit ("feat: ...", "test: ...", "bench: ..."). Don't move on with failing tests.
3. Keep the repo clean: no datasets or large files committed (add them to .gitignore; the README explains how to download SIFT1M), no dead code, no debug prints.
4. If a milestone is blocked (e.g. no network for the dataset download), skip to what can proceed, note it in the README under "Status", and keep going.
5. At the end, print a status table: milestone, done / partial / not done, and the exact commands to reproduce each result.

## Scope (in)
1. **Exact baseline:** brute-force search for ground truth and tests.
2. **HNSW core:** insert and search with parameters M, efConstruction, efSearch.
   - Level sampling with mL = 1/ln(M).
   - Layer 0 allows 2M neighbors.
   - The paper's neighbor-selection heuristic (Algorithm 4).
3. **Distance kernels:** L2 as scalar reference, then AVX2+FMA, tested equal to scalar within epsilon. Add inner product if time allows.
4. **Performance engineering:**
   - Contiguous per-node memory layout (links + vector together).
   - Epoch-tagged visited list (no clearing per query).
   - Software prefetch.
   - Allocation-free heaps in the hot path.
5. **Index file format:** header (magic, version, dim, M, counts, entry point), loaded with mmap so queries run directly on the mapped file.
6. **Concurrent search:** read-only graph, per-thread scratch state, thread pool, batch query API.
7. **Python bindings** (pybind11).
8. **Tests, benchmark harness, CI, README with results.**

## Scope (out, listed as future work in the README)
Concurrent inserts, deletes, filtered search, GPU, graphs larger than RAM beyond mmap, quantization.

## Milestones
- **M1:** CMake project, fvecs/ivecs readers, brute-force search, scalar L2, unit tests.
- **M2:** HNSW build and search (scalar). Test recall against brute force on small random datasets, plus graph invariants (degree limits, valid ids, entry point at the top layer).
- **M3:** AVX2 kernels with randomized tests against scalar, contiguous memory layout, epoch visited list, prefetch, allocation-free heaps.
- **M4:** mmap file format with save/load, round-trip test (identical results before and after), thread pool and batch query API.
- **M5:** Benchmark harness (see below), run on SIFT1M, recall-vs-QPS plot, results table.
- **M6:** pybind11 bindings with a small Python test and example.
- **M7:** sanitizer builds (ASan, UBSan) running the full test suite, a fuzz target for the index loader (libFuzzer) with a short saved run, GitHub Actions CI, README, and a short design doc in docs/.

## Benchmark methodology (M5)
- **Dataset:** SIFT1M (1,000,000 x 128-d float vectors, 10,000 queries, 100 ground-truth neighbors each). Write a script that downloads and prepares it. If the download fails, report that plainly, run the same harness on a synthetic dataset of the same shape, and label the results as synthetic.
- **Metric:** recall@10 vs queries per second, sweeping efSearch to trace the curve.
- **Baseline:** hnswlib built from source with identical flags (-O3 -march=native) and identical M and efConstruction (e.g. M=16, efC=200), on the same machine.
- **Report:** single-thread QPS, multi-thread QPS, build time, index size, peak memory, and load time (mmap vs read-into-memory).
- **Method:** warm-up runs, median of several runs, and the hardware, compiler, and flags recorded in the results file.
- **Ablation:** measure each optimization separately (memory layout, visited list, prefetch, SIMD) and report what each one contributed.
- **Honesty rule:** if the implementation is slower than hnswlib, say so in the README. If perf is available, include a short profile summary explaining where the time goes.

## Repo structure
```
include/        public headers
src/            library code
tests/          unit tests
bench/          benchmark, dataset, and plotting scripts; results/ for raw outputs
python/         pybind11 bindings
docs/           design doc
.github/        CI workflow
README.md       overview, build instructions, results table, recall-vs-QPS plot, reproduction commands, limitations, future work
CMakeLists.txt
```

## Final step
Write the README results section only from files in bench/results. Then print the three resume-bullet facts below, filled in from the saved results (and marked "not measured" where a result doesn't exist):
1. Headline: QPS at recall@10 on SIFT1M, and the gap to hnswlib.
2. Optimization: per-optimization speedups from the ablation.
3. Rigor or integration: test count, sanitizer status, fuzzing duration, CI status, and (if done) the Python bindings.