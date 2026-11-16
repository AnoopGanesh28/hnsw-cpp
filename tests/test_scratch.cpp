#include <limits>
#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw/scratch.h"
#include "hnsw_test_util.h"
#include "test_helper.h"

using namespace hnsw;

static KnnResult fresh_search(const HnswIndex& index, const float* q, size_t k, size_t ef) {
  SearchScratch scratch;
  return index.search(q, k, ef, scratch);
}

static bool same(const KnnResult& a, const KnnResult& b) {
  return a.ids == b.ids && a.dists == b.dists;
}

// One scratch reused for many queries gives the same results as a fresh one.
static void test_reuse_matches_fresh() {
  const size_t n = 800, dim = 10, nq = 60;
  const auto base = test::uniform_data(n, dim, 1);
  const auto queries = test::uniform_data(nq, dim, 2);
  const HnswIndex index = test::build(base, dim, test::params(8, 60, 3));
  SearchScratch reused;
  for (size_t round = 0; round < 3; ++round) {
    for (size_t q = 0; q < nq; ++q) {
      const float* qp = queries.data() + q * dim;
      const KnnResult want = fresh_search(index, qp, 10, 30);
      CHECK(same(index.search(qp, 10, 30, reused), want));
      CHECK(same(index.search(qp, 10, 30), want));  // thread_local scratch
    }
  }
}

// Drive the 16-bit epoch across its wraparound. The first pass on a fresh
// scratch runs at epoch 1 and leaves tag 1 on every node it visited. After
// wrapping, the epoch is 1 again, so without the clear-on-wrap those stale
// tags would read as "visited" and the repeated query would go wrong.
static void test_epoch_wraparound() {
  if (!kEpochVisited) return;
  const size_t n = 500, dim = 6;
  const auto base = test::uniform_data(n, dim, 4);
  const auto queries = test::uniform_data(4, dim, 5);
  const HnswIndex index = test::build(base, dim, test::params(6, 40, 6));
  const float* q0 = queries.data();
  const KnnResult want0 = fresh_search(index, q0, 10, 40);

  SearchScratch s;
  CHECK(same(index.search(q0, 10, 40, s), want0));
  CHECK_EQ(s.epoch(), SearchScratch::Tag{1});

  const SearchScratch::Tag max = std::numeric_limits<SearchScratch::Tag>::max();
  s.set_epoch_for_testing(static_cast<SearchScratch::Tag>(max - 2));
  for (size_t i = 1; i < 4; ++i) {  // epochs max-1, max, then wrap -> 1
    const float* qp = queries.data() + i * dim;
    CHECK(same(index.search(qp, 10, 40, s), fresh_search(index, qp, 10, 40)));
  }
  CHECK_EQ(s.epoch(), SearchScratch::Tag{1});
  CHECK(same(index.search(q0, 10, 40, s), want0));  // epoch 2, after the clear
  s.set_epoch_for_testing(max);
  CHECK(same(index.search(q0, 10, 40, s), want0));  // wraps to 1: stale tags from epoch 1
  CHECK_EQ(s.epoch(), SearchScratch::Tag{1});
}

// A scratch used on a small index grows when used on a larger one, and an
// index that grows after a search is handled by the same scratch.
static void test_scratch_grows_with_index() {
  const size_t dim = 5;
  const auto base = test::uniform_data(1200, dim, 7);
  const auto queries = test::uniform_data(20, dim, 8);
  SearchScratch s;
  HnswIndex index(dim, test::params(6, 40, 9));
  for (size_t i = 0; i < 1200; ++i) {
    index.add(base.data() + i * dim);
    if (i == 50 || i == 400 || i == 1199) {
      for (size_t q = 0; q < 20; ++q) {
        const float* qp = queries.data() + q * dim;
        CHECK(same(index.search(qp, 5, 20, s), fresh_search(index, qp, 5, 20)));
      }
    }
  }
  const HnswIndex small = test::build(test::uniform_data(30, dim, 10), dim, test::params(4, 10, 1));
  for (size_t q = 0; q < 20; ++q) {
    const float* qp = queries.data() + q * dim;
    CHECK(same(small.search(qp, 5, 20, s), fresh_search(small, qp, 5, 20)));
  }
}

int main() {
  test_reuse_matches_fresh();
  test_epoch_wraparound();
  test_scratch_grows_with_index();
  return test_finish();
}
