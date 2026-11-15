#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "hnsw/brute_force.h"
#include "hnsw/distance.h"

namespace hnsw {

struct HnswParams {
  // Max neighbors per node on layers >= 1; layer 0 allows 2 * M.
  size_t M = 16;
  // Candidate list size used while inserting.
  size_t ef_construction = 200;
  // Seed for level sampling. Same seed + same insertion order => same graph.
  uint64_t seed = 100;
  // Algorithm 4 keepPrunedConnections: fill up to M with discarded candidates.
  bool keep_pruned = false;
};

// Read-only view of one neighbor list.
struct NeighborView {
  const uint32_t* data = nullptr;
  size_t count = 0;
  const uint32_t* begin() const { return data; }
  const uint32_t* end() const { return data + count; }
};

// Hierarchical Navigable Small World index (Malkov & Yashunin), single
// threaded. Ids are assigned sequentially from 0 in insertion order.
class HnswIndex {
 public:
  // Throws std::invalid_argument if dim < 1, M < 2, ef_construction < 1, or
  // dist is null.
  HnswIndex(size_t dim, HnswParams params = {}, DistFn dist = l2_squared_scalar);

  // Copies dim floats from vec into the index and links it. Returns its id.
  uint32_t add(const float* vec);

  // Approximate k nearest neighbors, ascending by distance, ties by lower id.
  // Uses a candidate list of size max(ef, k). Returns at most min(k, size())
  // results; empty if the index is empty or k == 0.
  KnnResult search(const float* query, size_t k, size_t ef) const;

  size_t size() const { return levels_.size(); }
  size_t dim() const { return dim_; }
  const HnswParams& params() const { return params_; }

  // Highest layer in the graph; -1 when the index is empty.
  int max_level() const { return max_level_; }
  // Node searches start from. Throws std::logic_error when empty.
  uint32_t entry_point() const;
  // Top layer of node id. Throws std::out_of_range for an invalid id.
  int node_level(uint32_t id) const;
  // Neighbors of id on `level`. Throws std::out_of_range unless
  // id < size() and 0 <= level <= node_level(id).
  NeighborView neighbors(uint32_t id, int level) const;

  // Empty string if every structural invariant holds, otherwise a
  // description of the first violation found.
  std::string check_invariants() const;

 private:
  using Cand = std::pair<float, uint32_t>;  // (distance, id)

  const float* vec(uint32_t id) const { return vectors_.data() + size_t{id} * dim_; }
  float distance(const float* q, uint32_t id) const { return dist_(q, vec(id), dim_); }
  size_t cap(int level) const { return level == 0 ? 2 * params_.M : params_.M; }

  int sample_level();
  Cand greedy_closest(const float* q, Cand cur, int level) const;
  std::vector<Cand> search_layer(const float* q, const std::vector<Cand>& entry,
                                 size_t ef, int level) const;
  std::vector<uint32_t> select_neighbors(const std::vector<Cand>& sorted,
                                         size_t m) const;
  void connect(uint32_t from, uint32_t to, int level);

  size_t dim_;
  HnswParams params_;
  DistFn dist_;
  double ml_ = 0.0;
  std::mt19937_64 rng_;

  std::vector<float> vectors_;                           // row-major, size() * dim_
  std::vector<int> levels_;                              // top layer per node
  std::vector<std::vector<std::vector<uint32_t>>> links_;  // [node][level] -> neighbors
  int max_level_ = -1;
  uint32_t entry_ = 0;
};

}  // namespace hnsw
