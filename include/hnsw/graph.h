#pragma once

// GraphView, the read-only view of a graph that all search code runs on, and
// the search entry points over it (single query and batch).

#include <cstddef>
#include <cstdint>

#include "hnsw/brute_force.h"
#include "hnsw/distance.h"
#include "hnsw/layout.h"
#include "hnsw/scratch.h"

namespace hnsw {

class ThreadPool;

// Read-only view of one neighbor list.
struct NeighborView {
  const uint32_t* data = nullptr;
  size_t count = 0;
  const uint32_t* begin() const { return data; }
  const uint32_t* end() const { return data + count; }
};

// Read-only view of a complete graph. It only refers to memory owned
// elsewhere (an HnswIndex or, later, a mapped file) and is cheap to copy.
struct GraphView {
  Layout layout;
  const uint8_t* base = nullptr;            // count * record_stride bytes, 64-byte aligned
  size_t count = 0;
  const uint8_t* levels = nullptr;          // top layer of each node
  const uint64_t* upper_offsets = nullptr;  // per node, word offset into upper_links
  const uint32_t* upper_links = nullptr;
  uint32_t entry_point = 0;
  int max_level = -1;  // -1 when empty

  const uint8_t* record(uint32_t id) const { return base + layout.record_offset(id); }

  const float* vector(uint32_t id) const {
    return reinterpret_cast<const float*>(record(id) + layout.vector_offset);
  }

  // [count][ids...] block for node id on `level` (level <= levels[id]).
  const uint32_t* link_block(uint32_t id, int level) const {
    if (level == 0) {
      return reinterpret_cast<const uint32_t*>(record(id) + layout.count_offset);
    }
    return upper_links + layout.upper_block_offset(upper_offsets[id], level);
  }

  NeighborView neighbors(uint32_t id, int level) const {
    const uint32_t* block = link_block(id, level);
    return {block + 1, block[0]};
  }
};

// Approximate k-NN search (Algorithm 5) over a graph view. Results are
// ascending by distance, ties by lower id, at most min(k, count).
KnnResult search_graph(const GraphView& graph, DistFn dist, const float* query,
                       size_t k, size_t ef, SearchScratch& scratch);

// Same, writing into caller arrays of at least min(k, count) elements;
// returns the number of results. Performs no allocation once `scratch` has
// served a pass with this graph size and ef (when HNSW_OPT_EPOCH_VISITED and
// HNSW_OPT_ALLOC_FREE_HEAPS are on).
size_t search_graph_into(const GraphView& graph, DistFn dist, const float* query,
                         size_t k, size_t ef, SearchScratch& scratch, uint32_t* out_ids,
                         float* out_dists);

// Id written into batch result slots beyond the results found (fewer than k
// nodes reachable or in the index); its distance is +infinity. Never a valid
// id, since ids are below the node count, which fits in uint32.
inline constexpr uint32_t kNoNeighbor = UINT32_MAX;
// Queries per work chunk claimed by a pool thread, unless the caller says.
inline constexpr size_t kDefaultBatchChunk = 8;

// Searches n_queries row-major queries (n_queries * dim floats) on `pool`.
// Row q of out_ids / out_dists (n_queries * k elements each, row-major) gets
// query q's results, exactly as search_graph_into returns them, followed by
// kNoNeighbor / +infinity padding up to k. Each pool thread searches with its
// own pool.scratch(), so a warm pool performs no per-query allocation.
// Writes nothing if n_queries == 0 or k == 0. The graph must not change
// during the call. Throws std::invalid_argument if chunk == 0.
void search_graph_batch(const GraphView& graph, DistFn dist, const float* queries,
                        size_t n_queries, size_t k, size_t ef, uint32_t* out_ids,
                        float* out_dists, ThreadPool& pool, size_t chunk = kDefaultBatchChunk);

}  // namespace hnsw
