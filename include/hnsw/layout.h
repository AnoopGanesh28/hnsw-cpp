#pragma once

#include <cstddef>
#include <cstdint>

namespace hnsw {

// All storage arithmetic for the graph, derived from (dim, M) only. Every
// offset is relative to a base address, so the same layout describes an
// in-memory arena or a memory-mapped file (no pointers are ever stored).
//
// Layer-0 record for node id, at byte offset id * record_stride:
//   [count_offset ] uint32 link count
//   [links_offset ] max_links0 uint32 neighbor ids (first `count` valid)
//   [ ...padding  ] zero bytes up to vector_offset
//   [vector_offset] dim floats, 32-byte aligned
//   [ ...padding  ] zero bytes up to record_stride (a multiple of 64)
//
// Upper layers (level >= 1) live in a separate uint32 table. A node with
// level L >= 1 owns L consecutive blocks of upper_block words starting at its
// upper offset; block l - 1 holds layer l as [count][M ids].
//
// All values are stored little-endian (native on the supported x86-64 target).
struct Layout {
  static constexpr size_t kVectorAlign = 32;
  static constexpr size_t kRecordAlign = 64;

  size_t dim = 0;
  size_t M = 0;
  size_t max_links0 = 0;       // 2 * M
  size_t max_links_upper = 0;  // M
  size_t count_offset = 0;     // bytes
  size_t links_offset = 0;     // bytes
  size_t vector_offset = 0;    // bytes
  size_t record_stride = 0;    // bytes
  size_t upper_block = 0;      // uint32 words per upper layer: 1 + M

  static Layout make(size_t dim, size_t M) {
    Layout l;
    l.dim = dim;
    l.M = M;
    l.max_links0 = 2 * M;
    l.max_links_upper = M;
    l.count_offset = 0;
    l.links_offset = sizeof(uint32_t);
    l.vector_offset = round_up(l.links_offset + l.max_links0 * sizeof(uint32_t), kVectorAlign);
    l.record_stride = round_up(l.vector_offset + dim * sizeof(float), kRecordAlign);
    l.upper_block = 1 + M;
    return l;
  }

  size_t record_offset(uint32_t id) const { return size_t{id} * record_stride; }

  // Word offset into the upper table of the block for `level` (>= 1) of a
  // node whose blocks start at `node_upper_offset`.
  uint64_t upper_block_offset(uint64_t node_upper_offset, int level) const {
    return node_upper_offset + static_cast<uint64_t>(level - 1) * upper_block;
  }

  size_t max_links(int level) const { return level == 0 ? max_links0 : max_links_upper; }

  static constexpr size_t round_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
};

}  // namespace hnsw
