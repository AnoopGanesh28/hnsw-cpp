#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw_test_util.h"
#include "test_helper.h"

using namespace hnsw;

// Points 0, 1, ..., n-1 on a line, inserted in increasing order, with
// ef_construction >= n. For a new point k, every earlier point y on a layer is
// closer to k's predecessor p on that layer than to k (p - y < k - y), so the
// heuristic keeps only p. Every layer therefore becomes a doubly linked list:
// each node's neighbors are exactly its predecessor and successor on the layer.
static void test_line_becomes_linked_list() {
  const size_t n = 300;
  HnswIndex index(1, test::params(4, n, 9));
  for (size_t i = 0; i < n; ++i) {
    const float x = static_cast<float>(i);
    index.add(&x);
  }
  CHECK_EQ(index.check_invariants(), std::string());
  for (int l = 0; l <= index.max_level(); ++l) {
    std::vector<uint32_t> layer;  // nodes on layer l, ascending = by position
    for (uint32_t id = 0; id < n; ++id) {
      if (index.node_level(id) >= l) layer.push_back(id);
    }
    for (size_t i = 0; i < layer.size(); ++i) {
      std::vector<uint32_t> want;
      if (i > 0) want.push_back(layer[i - 1]);
      if (i + 1 < layer.size()) want.push_back(layer[i + 1]);
      const NeighborView nb = index.neighbors(layer[i], l);
      CHECK((std::vector<uint32_t>(nb.begin(), nb.end()) == want));
    }
  }
}

// keep_pruned refills the selection with discarded candidates in ascending
// distance order, so on the same line the last point gets its M nearest
// predecessors instead of one.
static void test_keep_pruned() {
  const size_t n = 50, M = 4;
  for (bool keep : {false, true}) {
    HnswParams p = test::params(M, n, 9);
    p.keep_pruned = keep;
    HnswIndex index(1, p);
    for (size_t i = 0; i < n; ++i) {
      const float x = static_cast<float>(i);
      index.add(&x);
    }
    const NeighborView nb = index.neighbors(static_cast<uint32_t>(n - 1), 0);
    const std::vector<uint32_t> got(nb.begin(), nb.end());
    const std::vector<uint32_t> want =
        keep ? std::vector<uint32_t>{48, 47, 46, 45} : std::vector<uint32_t>{48};
    CHECK(got == want);
    CHECK_EQ(index.check_invariants(), std::string());
  }
}

int main() {
  test_line_becomes_linked_list();
  test_keep_pruned();
  return test_finish();
}
