#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw/layout.h"
#include "hnsw_test_util.h"
#include "test_helper.h"

using namespace hnsw;

static void test_layout_arithmetic() {
  for (size_t dim : {1u, 3u, 7u, 8u, 16u, 31u, 128u, 960u}) {
    for (size_t M : {2u, 4u, 5u, 16u, 32u, 48u}) {
      const Layout l = Layout::make(dim, M);
      CHECK_EQ(l.max_links0, 2 * M);
      CHECK_EQ(l.max_links_upper, M);
      CHECK_EQ(l.count_offset, size_t{0});
      CHECK_EQ(l.links_offset, sizeof(uint32_t));
      CHECK(l.vector_offset >= l.links_offset + l.max_links0 * sizeof(uint32_t));
      CHECK_EQ(l.vector_offset % Layout::kVectorAlign, size_t{0});
      CHECK(l.vector_offset - (l.links_offset + l.max_links0 * sizeof(uint32_t)) <
            Layout::kVectorAlign);
      CHECK(l.record_stride >= l.vector_offset + dim * sizeof(float));
      CHECK_EQ(l.record_stride % Layout::kRecordAlign, size_t{0});
      CHECK(l.record_stride - (l.vector_offset + dim * sizeof(float)) < Layout::kRecordAlign);
      CHECK_EQ(l.upper_block, 1 + M);
      CHECK_EQ(l.upper_block_offset(100, 1), uint64_t{100});
      CHECK_EQ(l.upper_block_offset(100, 3), uint64_t{100 + 2 * (1 + M)});
    }
  }
  // The SIFT configuration: 4 + 32*4 = 132 -> vector at 160, 160 + 512 = 672 -> 704.
  const Layout sift = Layout::make(128, 16);
  CHECK_EQ(sift.vector_offset, size_t{160});
  CHECK_EQ(sift.record_stride, size_t{704});
}

static void test_arena_contents_and_alignment() {
  const size_t n = 300, dim = 13;
  const auto data = test::uniform_data(n, dim, 1);
  const HnswIndex index = test::build(data, dim, test::params(6, 40, 2));
  const GraphView g = index.view();
  CHECK_EQ(g.count, n);
  CHECK_EQ(reinterpret_cast<uintptr_t>(g.base) % Layout::kRecordAlign, uintptr_t{0});
  bool ok = true;
  for (uint32_t id = 0; id < n; ++id) {
    ok = ok && reinterpret_cast<uintptr_t>(g.vector(id)) % Layout::kVectorAlign == 0;
    ok = ok && std::memcmp(g.vector(id), data.data() + id * dim, dim * sizeof(float)) == 0;
    // Bytes between the links and the vector, and after the vector, are zero.
    const uint8_t* rec = g.record(id);
    const size_t links_end = g.layout.links_offset + g.layout.max_links0 * sizeof(uint32_t);
    for (size_t b = links_end; b < g.layout.vector_offset; ++b) ok = ok && rec[b] == 0;
    for (size_t b = g.layout.vector_offset + dim * sizeof(float); b < g.layout.record_stride; ++b) {
      ok = ok && rec[b] == 0;
    }
    // Unused link slots are zero.
    const uint32_t* block = g.link_block(id, 0);
    for (size_t s = 1 + block[0]; s <= g.layout.max_links0; ++s) ok = ok && block[s] == 0;
  }
  CHECK(ok);
  CHECK_EQ(index.check_invariants(), std::string());
}

static bool same_graph(const HnswIndex& a, const HnswIndex& b) {
  if (a.size() != b.size() || a.max_level() != b.max_level()) return false;
  for (uint32_t id = 0; id < a.size(); ++id) {
    if (a.node_level(id) != b.node_level(id)) return false;
    for (int l = 0; l <= a.node_level(id); ++l) {
      const NeighborView x = a.neighbors(id, l), y = b.neighbors(id, l);
      if (!std::equal(x.begin(), x.end(), y.begin(), y.end())) return false;
    }
  }
  return true;
}

// reserve() and amortized growth (with reallocation) must not affect the graph.
static void test_reserve_and_growth_equivalent() {
  const size_t n = 700, dim = 9;
  const auto data = test::uniform_data(n, dim, 3);
  const HnswParams p = test::params(5, 30, 4);
  HnswIndex grown(dim, p);
  HnswIndex reserved(dim, p);
  reserved.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    grown.add(data.data() + i * dim);
    reserved.add(data.data() + i * dim);
  }
  CHECK(same_graph(grown, reserved));
  const HnswIndex copy = grown;  // deep copy of the arena
  CHECK(same_graph(grown, copy));
  CHECK(copy.view().base != grown.view().base);
}

int main() {
  test_layout_arithmetic();
  test_arena_contents_and_alignment();
  test_reserve_and_growth_equivalent();
  return test_finish();
}
