#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw_test_util.h"
#include "test_helper.h"

using namespace hnsw;

// Checks every structural invariant through the public accessors only, so the
// test does not depend on (and independently verifies) check_invariants().
static void check_graph(const HnswIndex& index) {
  const size_t n = index.size();
  const size_t M = index.params().M;
  const int top = index.max_level();
  CHECK(top >= 0);
  CHECK_EQ(index.node_level(index.entry_point()), top);

  bool ok = true;
  for (uint32_t id = 0; id < n; ++id) {
    const int lvl = index.node_level(id);
    ok = ok && lvl >= 0 && lvl <= top;
    for (int l = 0; l <= lvl; ++l) {
      const NeighborView nb = index.neighbors(id, l);
      ok = ok && nb.count <= (l == 0 ? 2 * M : M);
      std::vector<uint32_t> sorted(nb.begin(), nb.end());
      std::sort(sorted.begin(), sorted.end());
      ok = ok && std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
      for (uint32_t x : nb) {
        ok = ok && x < n && x != id && index.node_level(x) >= l;
      }
    }
    // No neighbor list exists above the node's own level.
    CHECK_THROWS(index.neighbors(id, lvl + 1), std::out_of_range, "has no level");
    if (!ok) break;
  }
  CHECK(ok);
  CHECK_EQ(index.check_invariants(), std::string());
}

static void test_invariants_many_configs() {
  struct Config { size_t n, dim, M; uint64_t seed; bool keep_pruned; };
  const Config configs[] = {
      {1500, 3, 4, 1, false},  {1500, 8, 8, 2, false},  {1500, 24, 16, 3, false},
      {800, 16, 4, 4, true},   {800, 5, 2, 5, false},   {800, 12, 8, 6, true},
  };
  for (const Config& c : configs) {
    HnswParams p = test::params(c.M, 64, c.seed);
    p.keep_pruned = c.keep_pruned;
    const auto data = test::uniform_data(c.n, c.dim, static_cast<unsigned>(c.seed) + 100);
    const HnswIndex index = test::build(data, c.dim, p);
    CHECK_EQ(index.size(), c.n);
    check_graph(index);
  }
}

// P(level >= j) = P(-ln U >= j ln M) = M^-j exactly. The count of nodes with
// level >= j is Binomial(n, M^-j); allow 5 standard deviations (two-sided
// false-failure probability below 1e-6 under the normal approximation).
static void test_level_distribution() {
  const size_t n = 20000;
  for (size_t M : {4u, 16u}) {
    const auto data = test::uniform_data(n, 2, 7);
    const HnswIndex index = test::build(data, 2, test::params(M, 8, 42));
    for (int j : {1, 2}) {
      size_t at_least = 0;
      for (uint32_t id = 0; id < n; ++id) at_least += index.node_level(id) >= j;
      const double p = std::pow(static_cast<double>(M), -j);
      const double sigma = std::sqrt(static_cast<double>(n) * p * (1.0 - p));
      CHECK_NEAR(static_cast<double>(at_least), n * p, 5.0 * sigma);
    }
  }
}

int main() {
  test_invariants_many_configs();
  test_level_distribution();
  return test_finish();
}
