#pragma once

// Internal: layer-level search routines shared by insertion and k-NN search.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "hnsw/distance.h"
#include "hnsw/graph.h"
#include "hnsw/scratch.h"

namespace hnsw {
namespace detail {

using Cand = Candidate;

struct CandidateSpan {
  const Cand* data;
  size_t size;
};

// Greedy descent on one layer (SEARCH-LAYER with ef = 1): move to the closest
// neighbor while that improves the distance.
Cand greedy_closest(const GraphView& g, DistFn dist, const float* q, Cand cur, int level);

// Algorithm 2 (SEARCH-LAYER). Returns up to ef nearest found, ascending, in
// scratch.result_buffer() (valid until the next pass on this scratch).
// Visited marks come from `scratch` (epoch tags) or, with
// HNSW_OPT_EPOCH_VISITED off, from a fresh per-pass byte array. Heaps live in
// scratch buffers, or with HNSW_OPT_ALLOC_FREE_HEAPS off are
// std::priority_queue.
CandidateSpan search_layer(const GraphView& g, DistFn dist, const float* q,
                           const Cand* entry, size_t num_entry, size_t ef, int level,
                           SearchScratch& scratch);

}  // namespace detail
}  // namespace hnsw
