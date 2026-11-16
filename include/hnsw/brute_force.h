#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "hnsw/distance.h"

namespace hnsw {

// k nearest neighbors of one query: ids[i] and dists[i] describe the i-th
// nearest. Sorted ascending by distance; equal distances are ordered by lower
// id, so results are deterministic. dists holds whatever `dist` returns
// (squared L2 for l2_squared_scalar).
struct KnnResult {
  std::vector<uint32_t> ids;
  std::vector<float> dists;
};

// Exact kNN of `query` among n row-major vectors of dimension dim in `base`.
// Returns min(k, n) neighbors. Throws std::invalid_argument if n does not fit
// in a uint32_t id. The default kernel matches HnswIndex's default, so
// distances from both are directly comparable.
KnnResult brute_force_knn(const float* base, size_t n, size_t dim,
                          const float* query, size_t k,
                          DistFn dist = best_l2_squared());

// Same as brute_force_knn for nq row-major queries; result[i] belongs to
// query i. Single-threaded.
std::vector<KnnResult> brute_force_knn_batch(const float* base, size_t n,
                                             size_t dim, const float* queries,
                                             size_t nq, size_t k,
                                             DistFn dist = best_l2_squared());

// recall@k = |found ∩ truth| / k, using the first k entries of each list
// (fewer if a list is shorter). Duplicate ids in `found` count once. The
// divisor is always k, so a truth list shorter than k cannot reach 1.0.
// Throws std::invalid_argument if k == 0.
double recall_at_k(const std::vector<uint32_t>& found,
                   const std::vector<uint32_t>& truth, size_t k);

}  // namespace hnsw
