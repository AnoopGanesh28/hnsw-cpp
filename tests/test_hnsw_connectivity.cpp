#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw_test_util.h"
#include "test_helper.h"

using namespace hnsw;

// Every node must be reachable from the entry point on layer 0 for these
// configurations (M >= 8). Measured: full reachability on all of them.
// With M = 4 and M = 2 the heuristic (keep_pruned = false) was observed to
// leave a few nodes unreachable, so no full-reachability claim is made there.
static void test_full_reachability() {
  struct Config { size_t n, dim, M, efc; };
  const Config configs[] = {{2000, 16, 8, 200}, {2000, 16, 16, 200},
                            {2000, 32, 8, 200}, {2000, 32, 16, 200}};
  for (const Config& c : configs) {
    for (unsigned seed : {1u, 2u, 3u}) {
      const auto data = test::uniform_data(c.n, c.dim, seed);
      const HnswIndex index = test::build(data, c.dim, test::params(c.M, c.efc, seed));
      CHECK_EQ(test::reachable_on_layer0(index), c.n);
    }
  }
}

int main() {
  test_full_reachability();
  return test_finish();
}
