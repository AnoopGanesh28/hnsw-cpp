// Index built and searched with the AVX2 kernel versus the scalar kernel.
// Skips (exit 77) without AVX2 + FMA.

#include <algorithm>
#include <cmath>
#include <vector>

#include "hnsw/brute_force.h"
#include "hnsw/hnsw.h"
#include "hnsw_test_util.h"
#include "test_helper.h"

using namespace hnsw;

static bool same_graph(const HnswIndex& a, const HnswIndex& b) {
  if (a.size() != b.size() || a.max_level() != b.max_level()) return false;
  for (uint32_t id = 0; id < a.size(); ++id) {
    for (int l = 0; l <= a.node_level(id); ++l) {
      const NeighborView x = a.neighbors(id, l), y = b.neighbors(id, l);
      if (!std::equal(x.begin(), x.end(), y.begin(), y.end())) return false;
    }
  }
  return true;
}

// Rounding differences between kernels may change a few links, so graphs are
// not required to match. Stated tolerance: mean recall@10 within 0.005
// (10 of 2000 hits) of the scalar-kernel index at every ef. Measured on these
// configurations (seeds 1-5, dims 16/32/128): identical to 4 decimals.
static void test_recall_matches_scalar() {
  const size_t n = 2000, nq = 200, k = 10;
  for (size_t dim : {16u, 32u, 128u}) {
    for (unsigned seed : {1u, 2u}) {
      const auto base = test::uniform_data(n, dim, seed);
      const auto queries = test::uniform_data(nq, dim, seed + 1000);
      const auto truth =
          brute_force_knn_batch(base.data(), n, dim, queries.data(), nq, k, l2_squared_scalar);
      HnswIndex scalar(dim, test::params(16, 200, seed), l2_squared_scalar);
      HnswIndex simd(dim, test::params(16, 200, seed), l2_squared_avx2);
      for (size_t i = 0; i < n; ++i) {
        scalar.add(base.data() + i * dim);
        simd.add(base.data() + i * dim);
      }
      CHECK_EQ(simd.check_invariants(), std::string());
      for (size_t ef : {10u, 40u, 80u}) {
        CHECK_NEAR(test::mean_recall(simd, queries, truth, k, ef),
                   test::mean_recall(scalar, queries, truth, k, ef), 0.005);
      }
    }
  }
}

// The default kernel is the dispatched one: same graph as an explicit AVX2 index.
static void test_default_kernel_is_avx2() {
  const size_t n = 500, dim = 20;
  const auto base = test::uniform_data(n, dim, 9);
  const HnswIndex def = test::build(base, dim, test::params(8, 50, 1));
  HnswIndex explicit_avx2(dim, test::params(8, 50, 1), l2_squared_avx2);
  for (size_t i = 0; i < n; ++i) explicit_avx2.add(base.data() + i * dim);
  CHECK(same_graph(def, explicit_avx2));
}

int main() {
  if (!cpu_supports_avx2_fma()) return 77;
  test_recall_matches_scalar();
  test_default_kernel_is_avx2();
  return test_finish();
}
