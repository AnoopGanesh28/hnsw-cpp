#pragma once

#include <cstddef>
#include <cstdint>

#include "hnsw/brute_force.h"
#include "hnsw/distance.h"
#include "hnsw/layout.h"

namespace hnsw {

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
                       size_t k, size_t ef);

}  // namespace hnsw
