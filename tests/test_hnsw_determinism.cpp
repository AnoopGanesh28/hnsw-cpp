#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw_test_util.h"
#include "test_helper.h"

using namespace hnsw;

static bool same_graph(const HnswIndex& a, const HnswIndex& b) {
  if (a.size() != b.size() || a.max_level() != b.max_level()) return false;
  if (a.size() > 0 && a.entry_point() != b.entry_point()) return false;
  for (uint32_t id = 0; id < a.size(); ++id) {
    if (a.node_level(id) != b.node_level(id)) return false;
    for (int l = 0; l <= a.node_level(id); ++l) {
      const NeighborView x = a.neighbors(id, l), y = b.neighbors(id, l);
      if (std::vector<uint32_t>(x.begin(), x.end()) !=
          std::vector<uint32_t>(y.begin(), y.end())) {
        return false;
      }
    }
  }
  return true;
}

static void test_same_seed_same_graph_and_results() {
  const size_t n = 1000, dim = 12, nq = 30;
  const auto base = test::uniform_data(n, dim, 1);
  const auto queries = test::uniform_data(nq, dim, 2);
  const HnswIndex a = test::build(base, dim, test::params(8, 64, 123));
  const HnswIndex b = test::build(base, dim, test::params(8, 64, 123));
  CHECK(same_graph(a, b));
  for (size_t q = 0; q < nq; ++q) {
    const KnnResult ra = a.search(queries.data() + q * dim, 10, 20);
    const KnnResult rb = b.search(queries.data() + q * dim, 10, 20);
    CHECK(ra.ids == rb.ids);
    CHECK(ra.dists == rb.dists);
  }
}

static void test_different_seed_different_graph() {
  const size_t n = 1000, dim = 12;
  const auto base = test::uniform_data(n, dim, 1);
  const HnswIndex a = test::build(base, dim, test::params(8, 64, 123));
  const HnswIndex b = test::build(base, dim, test::params(8, 64, 124));
  CHECK(!same_graph(a, b));
  size_t level_diffs = 0;
  for (uint32_t id = 0; id < n; ++id) level_diffs += a.node_level(id) != b.node_level(id);
  CHECK(level_diffs > 0);
}

int main() {
  test_same_seed_same_graph_and_results();
  test_different_seed_different_graph();
  return test_finish();
}
