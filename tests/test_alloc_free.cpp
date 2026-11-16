// Proves search_into performs no heap allocation once its scratch is warm.
// Every global operator new form is replaced with a counting version.

#include <vector>

#include "alloc_hook.h"
#include "hnsw/config.h"
#include "hnsw/hnsw.h"
#include "hnsw_test_util.h"
#include "test_helper.h"

using test::g_allocations;

using namespace hnsw;

static void test_counter_works() {
  const long before = g_allocations;
  std::vector<int>* v = new std::vector<int>(10);
  delete v;
  CHECK(g_allocations - before >= 2);
}

static void test_search_into_allocates_nothing_when_warm() {
  const size_t n = 2000, dim = 24, nq = 100, k = 10;
  const auto base = test::uniform_data(n, dim, 1);
  const auto queries = test::uniform_data(nq, dim, 2);
  const HnswIndex index = test::build(base, dim, test::params(12, 80, 3));
  SearchScratch scratch;
  std::vector<uint32_t> ids(k);
  std::vector<float> dists(k);
  const size_t efs[] = {10, 50, 200};

  for (size_t ef : efs) {  // warm-up: grows scratch to the largest ef used
    for (size_t q = 0; q < nq; ++q) {
      index.search_into(queries.data() + q * dim, k, ef, scratch, ids.data(), dists.data());
    }
  }
  const long before = g_allocations;
  size_t results = 0;
  for (int round = 0; round < 3; ++round) {
    for (size_t ef : efs) {
      for (size_t q = 0; q < nq; ++q) {
        results += index.search_into(queries.data() + q * dim, k, ef, scratch, ids.data(),
                                     dists.data());
      }
    }
  }
  CHECK_EQ(g_allocations - before, 0L);
  CHECK_EQ(results, size_t{3 * 3 * nq * k});
}

// search_into and search return identical results.
static void test_matches_search() {
  const size_t n = 500, dim = 8;
  const auto base = test::uniform_data(n, dim, 4);
  const auto queries = test::uniform_data(30, dim, 5);
  const HnswIndex index = test::build(base, dim, test::params(6, 40, 6));
  SearchScratch scratch;
  for (size_t q = 0; q < 30; ++q) {
    const float* qp = queries.data() + q * dim;
    for (size_t k : {1u, 7u, 600u}) {
      std::vector<uint32_t> ids(std::min(k, n));
      std::vector<float> dists(ids.size());
      const size_t got = index.search_into(qp, k, 20, scratch, ids.data(), dists.data());
      ids.resize(got);
      dists.resize(got);
      const KnnResult want = index.search(qp, k, 20);
      CHECK(ids == want.ids);
      CHECK(dists == want.dists);
    }
  }
}

int main() {
  test_counter_works();
  test_matches_search();
  if (!kEpochVisited || !kAllocFreeHeaps) {
    // The zero-allocation claim only holds with both optimizations on.
    return test_finish() != 0 ? 1 : 77;
  }
  test_search_into_allocates_nothing_when_warm();
  return test_finish();
}
