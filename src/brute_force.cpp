#include "hnsw/brute_force.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace hnsw {

KnnResult brute_force_knn(const float* base, size_t n, size_t dim,
                          const float* query, size_t k, DistFn dist) {
  if (n > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("brute_force_knn: n does not fit in uint32 ids");
  }
  std::vector<std::pair<float, uint32_t>> all(n);
  for (size_t i = 0; i < n; ++i) {
    all[i] = {dist(base + i * dim, query, dim), static_cast<uint32_t>(i)};
  }
  // pair ordering is (distance, id): ties resolve to the lower id.
  const size_t take = std::min(k, n);
  std::partial_sort(all.begin(), all.begin() + take, all.end());

  KnnResult result;
  result.ids.reserve(take);
  result.dists.reserve(take);
  for (size_t i = 0; i < take; ++i) {
    result.dists.push_back(all[i].first);
    result.ids.push_back(all[i].second);
  }
  return result;
}

std::vector<KnnResult> brute_force_knn_batch(const float* base, size_t n,
                                             size_t dim, const float* queries,
                                             size_t nq, size_t k,
                                             DistFn dist) {
  std::vector<KnnResult> results;
  results.reserve(nq);
  for (size_t q = 0; q < nq; ++q) {
    results.push_back(brute_force_knn(base, n, dim, queries + q * dim, k, dist));
  }
  return results;
}

double recall_at_k(const std::vector<uint32_t>& found,
                   const std::vector<uint32_t>& truth, size_t k) {
  if (k == 0) {
    throw std::invalid_argument("recall_at_k: k must be positive");
  }
  std::vector<uint32_t> f(found.begin(), found.begin() + std::min(k, found.size()));
  std::vector<uint32_t> t(truth.begin(), truth.begin() + std::min(k, truth.size()));
  std::sort(f.begin(), f.end());
  f.erase(std::unique(f.begin(), f.end()), f.end());
  std::sort(t.begin(), t.end());
  t.erase(std::unique(t.begin(), t.end()), t.end());

  std::vector<uint32_t> common;
  std::set_intersection(f.begin(), f.end(), t.begin(), t.end(),
                        std::back_inserter(common));
  return static_cast<double>(common.size()) / static_cast<double>(k);
}

}  // namespace hnsw
