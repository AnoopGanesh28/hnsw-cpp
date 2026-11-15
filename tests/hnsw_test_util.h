#pragma once

// Shared helpers for HNSW tests.

#include <cstdint>
#include <random>
#include <vector>

#include "hnsw/brute_force.h"
#include "hnsw/hnsw.h"

namespace test {

inline std::vector<float> uniform_data(size_t n, size_t dim, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::vector<float> v(n * dim);
  for (float& x : v) x = u(rng);
  return v;
}

inline hnsw::HnswIndex build(const std::vector<float>& data, size_t dim,
                             const hnsw::HnswParams& params) {
  hnsw::HnswIndex index(dim, params);
  for (size_t i = 0; i < data.size() / dim; ++i) index.add(data.data() + i * dim);
  return index;
}

inline hnsw::HnswParams params(size_t M, size_t ef_construction, uint64_t seed) {
  hnsw::HnswParams p;
  p.M = M;
  p.ef_construction = ef_construction;
  p.seed = seed;
  return p;
}

// Number of nodes reachable from the entry point following layer-0 links.
inline size_t reachable_on_layer0(const hnsw::HnswIndex& index) {
  if (index.size() == 0) return 0;
  std::vector<uint8_t> seen(index.size(), 0);
  std::vector<uint32_t> stack{index.entry_point()};
  seen[index.entry_point()] = 1;
  size_t count = 1;
  while (!stack.empty()) {
    const uint32_t x = stack.back();
    stack.pop_back();
    for (uint32_t y : index.neighbors(x, 0)) {
      if (!seen[y]) {
        seen[y] = 1;
        ++count;
        stack.push_back(y);
      }
    }
  }
  return count;
}

// Mean recall@k of index.search(..., k, ef) against exact ground truth.
inline double mean_recall(const hnsw::HnswIndex& index, const std::vector<float>& queries,
                          const std::vector<hnsw::KnnResult>& truth, size_t k, size_t ef) {
  const size_t dim = index.dim();
  double sum = 0.0;
  for (size_t q = 0; q < truth.size(); ++q) {
    sum += hnsw::recall_at_k(index.search(queries.data() + q * dim, k, ef).ids,
                             truth[q].ids, k);
  }
  return sum / static_cast<double>(truth.size());
}

}  // namespace test
