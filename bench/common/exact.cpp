#include "exact.h"

namespace bench {

std::vector<hnsw::KnnResult> exact_knn(const float* base, size_t n, size_t dim,
                                       const float* queries,
                                       const std::vector<size_t>& query_ids, size_t k,
                                       hnsw::ThreadPool& pool) {
  std::vector<hnsw::KnnResult> out(query_ids.size());
  pool.parallel_for(query_ids.size(), 1, [&](size_t, size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      out[i] = hnsw::brute_force_knn(base, n, dim, queries + query_ids[i] * dim, k);
    }
  });
  return out;
}

}  // namespace bench
