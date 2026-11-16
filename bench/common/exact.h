#pragma once

#include <cstddef>
#include <vector>

#include "hnsw/brute_force.h"
#include "hnsw/thread_pool.h"

namespace bench {

// Exact k nearest neighbors of queries[query_ids[i]] among the n base vectors,
// for every i, computed with hnsw::brute_force_knn (default kernel) in
// parallel on `pool`, one query per work item. result[i] belongs to
// query_ids[i]; order and ties follow brute_force_knn.
std::vector<hnsw::KnnResult> exact_knn(const float* base, size_t n, size_t dim,
                                       const float* queries,
                                       const std::vector<size_t>& query_ids, size_t k,
                                       hnsw::ThreadPool& pool);

}  // namespace bench
