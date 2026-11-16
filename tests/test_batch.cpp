// search_batch on HnswIndex and LoadedIndex equals sequential single-query
// search bit for bit, for any thread count and chunk size, with padding when
// fewer than k results exist.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "alloc_hook.h"
#include "hnsw/config.h"
#include "hnsw/hnsw.h"
#include "hnsw/index_file.h"
#include "hnsw/thread_pool.h"
#include "index_file_util.h"
#include "test_helper.h"

using namespace hnsw;

namespace {

struct Rows {
  std::vector<uint32_t> ids;
  std::vector<float> dists;
  bool operator==(const Rows& o) const {
    return ids == o.ids && dists.size() == o.dists.size() &&
           (dists.empty() ||
            std::memcmp(dists.data(), o.dists.data(), dists.size() * sizeof(float)) == 0);
  }
};

// Expected batch output built from one-query-at-a-time search().
template <typename Index>
Rows sequential(const Index& index, const std::vector<float>& queries, size_t nq, size_t k,
                size_t ef) {
  Rows r{std::vector<uint32_t>(nq * k, kNoNeighbor),
         std::vector<float>(nq * k, std::numeric_limits<float>::infinity())};
  SearchScratch scratch;
  for (size_t q = 0; q < nq; ++q) {
    const KnnResult res = index.search(queries.data() + q * index.dim(), k, ef, scratch);
    std::copy(res.ids.begin(), res.ids.end(), r.ids.begin() + q * k);
    std::copy(res.dists.begin(), res.dists.end(), r.dists.begin() + q * k);
  }
  return r;
}

template <typename Index>
Rows batch(const Index& index, const std::vector<float>& queries, size_t nq, size_t k, size_t ef,
           ThreadPool& pool, size_t chunk) {
  // Prefilled with values the batch must overwrite.
  Rows r{std::vector<uint32_t>(nq * k, 12345u), std::vector<float>(nq * k, -1.0f)};
  index.search_batch(queries.data(), nq, k, ef, r.ids.data(), r.dists.data(), pool, chunk);
  return r;
}

}  // namespace

static void test_matches_sequential() {
  const size_t n = 1500, dim = 16, nq = 203;
  const HnswIndex index = test::build_exact(n, dim, 8, 60, 21, best_l2_squared());
  const test::TempFile file("batch");
  index.save(file.path());
  const LoadedIndex loaded = LoadedIndex::open(file.path());
  const std::vector<float> queries = test::exact_data(nq, dim, 77);
  const std::pair<size_t, size_t> kef[] = {{10, 32}, {1, 10}, {50, 50}};
  for (const auto& [k, ef] : kef) {
    const Rows want = sequential(index, queries, nq, k, ef);
    CHECK(want == sequential(loaded, queries, nq, k, ef));
    for (size_t threads : {1u, 2u, 4u, 8u}) {
      ThreadPool pool(threads);
      for (size_t chunk : {size_t{1}, size_t{5}, kDefaultBatchChunk, size_t{1000}}) {  // 1000 > nq
        CHECK(batch(index, queries, nq, k, ef, pool, chunk) == want);
        CHECK(batch(loaded, queries, nq, k, ef, pool, chunk) == want);
      }
    }
  }
}

static void test_edge_cases() {
  const size_t dim = 7;
  const HnswIndex index = test::build_exact(300, dim, 6, 40, 4, best_l2_squared());
  const std::vector<float> queries = test::exact_data(5, dim, 8);
  ThreadPool pool(8);

  // n_queries = 0 and k = 0 write nothing.
  std::vector<uint32_t> ids(4, 7u);
  std::vector<float> dists(4, 7.0f);
  index.search_batch(queries.data(), 0, 2, 10, ids.data(), dists.data(), pool);
  index.search_batch(queries.data(), 2, 0, 10, ids.data(), dists.data(), pool);
  CHECK(ids == std::vector<uint32_t>(4, 7u));
  CHECK(dists == std::vector<float>(4, 7.0f));
  CHECK_THROWS(index.search_batch(queries.data(), 2, 2, 10, ids.data(), dists.data(), pool, 0),
               std::invalid_argument, "chunk");

  // Fewer queries than threads.
  CHECK(batch(index, queries, 3, 4, 20, pool, 1) == sequential(index, queries, 3, 4, 20));

  // k > n: every row holds all n nodes, then padding.
  const HnswIndex tiny = test::build_exact(5, dim, 4, 10, 9, best_l2_squared());
  const Rows got = batch(tiny, queries, 5, 8, 16, pool, 2);
  CHECK(got == sequential(tiny, queries, 5, 8, 16));
  bool padded = true;
  for (size_t q = 0; q < 5; ++q) {
    for (size_t i = 5; i < 8; ++i) {
      padded = padded && got.ids[q * 8 + i] == kNoNeighbor &&
               std::isinf(got.dists[q * 8 + i]) && got.dists[q * 8 + i] > 0;
    }
    for (size_t i = 0; i < 5; ++i) padded = padded && got.ids[q * 8 + i] < 5;
  }
  CHECK(padded);

  // Empty index: all padding.
  const HnswIndex empty(dim);
  const Rows none = batch(empty, queries, 4, 3, 10, pool, 1);
  CHECK(none.ids == std::vector<uint32_t>(12, kNoNeighbor));
  bool all_inf = true;
  for (float d : none.dists) all_inf = all_inf && std::isinf(d) && d > 0;
  CHECK(all_inf);
}

// A warm pool allocates nothing per query: the per-batch allocation count
// does not grow with the number of queries.
static void test_no_per_query_allocation() {
  if (!kEpochVisited || !kAllocFreeHeaps) return;  // claim needs both optimizations
  const size_t dim = 12, k = 10, ef = 40;
  const HnswIndex index = test::build_exact(2000, dim, 8, 40, 31, best_l2_squared());
  const std::vector<float> queries = test::exact_data(400, dim, 32);
  std::vector<uint32_t> ids(400 * k);
  std::vector<float> dists(400 * k);
  ThreadPool pool(4);
  for (size_t t = 0; t < pool.size(); ++t) {  // warm every thread's scratch
    index.search_into(queries.data(), k, ef, pool.scratch(t), ids.data(), dists.data());
  }
  index.search_batch(queries.data(), 400, k, ef, ids.data(), dists.data(), pool, 1);
  auto allocs = [&](size_t nq) {
    const long before = test::g_allocations;
    index.search_batch(queries.data(), nq, k, ef, ids.data(), dists.data(), pool, 1);
    return test::g_allocations - before;
  };
  const long small = allocs(4), large = allocs(400);
  CHECK(large <= small);
  CHECK(small <= 1);  // at most the std::function holding the task
}

int main() {
  test_matches_sequential();
  test_edge_cases();
  test_no_per_query_allocation();
  return test_finish();
}
