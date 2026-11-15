#include <vector>

#include "hnsw/brute_force.h"
#include "hnsw/hnsw.h"
#include "hnsw_test_util.h"
#include "test_helper.h"

using namespace hnsw;

// Uniform random data, n = 2000, 200 queries, ef_construction = 200.
// Thresholds were set from measurements over seeds 1-5 (minimum observed
// recall@10: dim 16/M 16/ef 40 -> 0.9965; dim 32/M 16/ef 80 -> 0.9980;
// dim 16/M 8/ef 40 -> 0.9815) with a margin below the minimum.
struct Case { size_t dim, M, ef; double min_recall; };
const Case kCases[] = {{16, 16, 40, 0.98}, {32, 16, 80, 0.98}, {16, 8, 40, 0.95}};
const size_t kN = 2000, kQueries = 200, kK = 10;

static void test_recall_threshold_and_monotone() {
  for (const Case& c : kCases) {
    for (unsigned seed : {1u, 2u, 3u}) {
      const auto base = test::uniform_data(kN, c.dim, seed);
      const auto queries = test::uniform_data(kQueries, c.dim, seed + 1000);
      const HnswIndex index = test::build(base, c.dim, test::params(c.M, 200, seed));
      const auto truth = brute_force_knn_batch(base.data(), kN, c.dim, queries.data(),
                                               kQueries, kK);
      CHECK(test::mean_recall(index, queries, truth, kK, c.ef) >= c.min_recall);

      // Mean recall must not drop as ef grows (0.002 slack = 4 of 2000 hits).
      double prev = 0.0;
      for (size_t ef : {10u, 20u, 40u, 80u, 160u, 320u}) {
        const double r = test::mean_recall(index, queries, truth, kK, ef);
        CHECK(r >= prev - 0.002);
        prev = r;
      }
    }
  }
}

// With ef >= n, SEARCH-LAYER keeps every visited node, so it explores the whole
// layer-0 component of the entry point. These graphs are fully reachable (see
// test_hnsw_connectivity), so results must equal brute force exactly.
static void test_ef_n_equals_brute_force() {
  const size_t n = 500, dim = 16, nq = 50;
  for (size_t M : {8u, 16u}) {
    const auto base = test::uniform_data(n, dim, 77);
    const auto queries = test::uniform_data(nq, dim, 78);
    const HnswIndex index = test::build(base, dim, test::params(M, 100, 5));
    for (size_t q = 0; q < nq; ++q) {
      const float* qp = queries.data() + q * dim;
      for (size_t k : {1u, 10u, 100u}) {
        const KnnResult got = index.search(qp, k, n);
        const KnnResult want = brute_force_knn(base.data(), n, dim, qp, k);
        CHECK(got.ids == want.ids);
        CHECK(got.dists == want.dists);
      }
    }
  }
}

int main() {
  test_recall_threshold_and_monotone();
  test_ef_n_equals_brute_force();
  return test_finish();
}
