"""Build an index, search it, save and reload it, and compare with exact search."""

import os
import tempfile

import numpy as np

import hnsw_cpp

DIM, N, NQ, K, EF = 64, 5000, 100, 10, 100

rng = np.random.default_rng(42)
base = rng.random((N, DIM), dtype=np.float32)
queries = rng.random((NQ, DIM), dtype=np.float32)

index = hnsw_cpp.Index(DIM, M=16, ef_construction=200)
ids = index.add(base)
print(f"added {len(ids)} vectors, index.size = {index.size}")

found, sq_dists = index.search(queries, K, EF)  # distances are SQUARED L2
print(f"query 0: ids {found[0][:3].tolist()}, squared distances {sq_dists[0][:3].tolist()}")

# Exact search in float64 for comparison.
exact = np.argsort(
    ((queries.astype(np.float64)[:, None, :] - base.astype(np.float64)[None]) ** 2).sum(-1),
    axis=1,
)[:, :K]
recall = np.mean([len(set(a) & set(b)) / K for a, b in zip(found, exact)])
print(f"recall@{K} vs exact brute force: {recall:.4f}")

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "index.hnsw")
    index.save(path)
    with hnsw_cpp.load(path, mode="mmap", validate="deep") as loaded:
        found2, dists2 = loaded.search(queries, K, EF, threads=0)
        identical = np.array_equal(found, found2) and sq_dists.tobytes() == dists2.tobytes()
        print(f"loaded index ({loaded.size} x {loaded.dim}) gives identical results: {identical}")
        assert identical
