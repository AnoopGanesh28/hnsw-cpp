#pragma once

// Internal: layer-level search routines shared by insertion and k-NN search.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "hnsw/distance.h"
#include "hnsw/graph.h"

namespace hnsw {
namespace detail {

// (distance, id). Compared by distance, then id: a total order, so traversal
// and results are deterministic under distance ties.
using Cand = std::pair<float, uint32_t>;

// Greedy descent on one layer (SEARCH-LAYER with ef = 1): move to the closest
// neighbor while that improves the distance.
Cand greedy_closest(const GraphView& g, DistFn dist, const float* q, Cand cur, int level);

// Algorithm 2 (SEARCH-LAYER). Returns up to ef nearest found, ascending.
std::vector<Cand> search_layer(const GraphView& g, DistFn dist, const float* q,
                               const std::vector<Cand>& entry, size_t ef, int level);

}  // namespace detail
}  // namespace hnsw
