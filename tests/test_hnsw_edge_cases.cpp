#include <limits>
#include <stdexcept>
#include <vector>

#include "hnsw/brute_force.h"
#include "hnsw/hnsw.h"
#include "hnsw/index_file.h"
#include "hnsw_test_util.h"
#include "index_file_util.h"
#include "test_helper.h"

using namespace hnsw;

static void test_constructor_validation() {
  CHECK_THROWS(HnswIndex(0), std::invalid_argument, "dim must be >= 1");
  CHECK_THROWS(HnswIndex(4, test::params(0, 10, 1)), std::invalid_argument, "M must be >= 2");
  CHECK_THROWS(HnswIndex(4, test::params(1, 10, 1)), std::invalid_argument, "M must be >= 2");
  CHECK_THROWS(HnswIndex(4, test::params(2, 0, 1)), std::invalid_argument, "ef_construction");
  CHECK_THROWS(HnswIndex(4, HnswParams{}, nullptr), std::invalid_argument, "distance function");
  HnswIndex ok(4, test::params(2, 1, 1));
  CHECK_EQ(ok.dim(), size_t{4});
}

static void test_empty_index() {
  const HnswIndex index(3);
  const float q[] = {0, 0, 0};
  CHECK_EQ(index.size(), size_t{0});
  CHECK_EQ(index.max_level(), -1);
  CHECK(index.search(q, 5, 10).ids.empty());
  CHECK(index.search(q, 5, 10).dists.empty());
  CHECK_THROWS(index.entry_point(), std::logic_error, "empty");
  CHECK_THROWS(index.node_level(0), std::out_of_range, "out of range");
  CHECK_EQ(index.check_invariants(), std::string());
}

static void test_one_and_two_elements() {
  HnswIndex index(2);
  const float a[] = {1, 1}, b[] = {4, 5}, q[] = {0, 0};
  CHECK_EQ(index.add(a), 0u);
  CHECK_EQ(index.entry_point(), 0u);
  CHECK_EQ(index.max_level(), index.node_level(0));
  KnnResult r = index.search(q, 3, 10);
  CHECK((r.ids == std::vector<uint32_t>{0}));
  CHECK((r.dists == std::vector<float>{2.0f}));

  CHECK_EQ(index.add(b), 1u);
  CHECK_EQ(index.check_invariants(), std::string());
  CHECK_EQ(index.neighbors(0, 0).count, size_t{1});
  CHECK_EQ(index.neighbors(1, 0).count, size_t{1});
  r = index.search(q, 2, 1);
  CHECK((r.ids == std::vector<uint32_t>{0, 1}));
  CHECK((r.dists == std::vector<float>{2.0f, 41.0f}));
  CHECK(index.search(q, 0, 10).ids.empty());
}

static void test_k_variants() {
  const size_t n = 60, dim = 4;
  const auto base = test::uniform_data(n, dim, 3);
  const auto queries = test::uniform_data(10, dim, 4);
  const HnswIndex index = test::build(base, dim, test::params(16, 100, 1));
  CHECK_EQ(test::reachable_on_layer0(index), n);
  for (size_t q = 0; q < 10; ++q) {
    const float* qp = queries.data() + q * dim;
    // k > n: effective ef = k >= n explores everything, returns all n.
    const KnnResult all = index.search(qp, n + 40, 1);
    CHECK_EQ(all.ids.size(), n);
    CHECK(all.ids == brute_force_knn(base.data(), n, dim, qp, n).ids);
    // ef < k: effective ef is raised to k.
    CHECK_EQ(index.search(qp, 10, 1).ids.size(), size_t{10});
    // k = 1 with a generous ef finds the exact nearest.
    CHECK(index.search(qp, 1, n).ids == brute_force_knn(base.data(), n, dim, qp, 1).ids);
  }
}

static std::vector<float> with_duplicates(size_t random_n, size_t copies,
                                          const std::vector<float>& dup) {
  std::vector<float> base = test::uniform_data(random_n, dup.size(), 5);
  for (size_t i = 0; i < copies; ++i) base.insert(base.end(), dup.begin(), dup.end());
  return base;
}

static void test_duplicates() {
  const size_t dim = 3;
  const std::vector<float> dup = {0.5f, 0.5f, 0.5f};

  // A few duplicates (fewer than M): graph stays connected and exact search
  // with ef >= n matches brute force, including the lower-id tie order.
  {
    const std::vector<float> base = with_duplicates(30, 3, dup);
    const size_t n = base.size() / dim;
    const HnswIndex index = test::build(base, dim, test::params(8, 50, 2));
    CHECK_EQ(index.check_invariants(), std::string());
    CHECK_EQ(test::reachable_on_layer0(index), n);
    const KnnResult r = index.search(dup.data(), 5, n);
    CHECK(r.ids == brute_force_knn(base.data(), n, dim, dup.data(), 5).ids);
  }

  // Many exact duplicates (30 copies, M = 4). Known HNSW limitation, observed
  // here: later copies all link to the lowest-id copies, whose lists are full
  // of distance-0 ties, so the shrink step drops the links back to the later
  // copies and they become unreachable on layer 0. The graph is still valid
  // and every returned neighbor is an exact match, but fewer than k may be
  // returned, so no stronger claim is made.
  {
    const std::vector<float> base = with_duplicates(30, 30, dup);
    const size_t n = base.size() / dim;
    const HnswIndex index = test::build(base, dim, test::params(4, 50, 2));
    CHECK_EQ(index.check_invariants(), std::string());
    const KnnResult r = index.search(dup.data(), 10, n);
    CHECK(!r.ids.empty());
    for (float d : r.dists) CHECK_EQ(d, 0.0f);
  }

  // All vectors identical.
  const std::vector<float> same(40 * dim, 2.0f);
  const HnswIndex flat = test::build(same, dim, test::params(4, 20, 3));
  CHECK_EQ(flat.check_invariants(), std::string());
  const float q[] = {2.0f, 2.0f, 2.0f};
  const KnnResult s = flat.search(q, 5, 40);
  CHECK_EQ(s.ids.size(), size_t{5});
  for (float d : s.dists) CHECK_EQ(d, 0.0f);
}

static void test_dim_one() {
  const size_t n = 200;
  const auto base = test::uniform_data(n, 1, 6);
  const HnswIndex index = test::build(base, 1, test::params(8, 50, 4));
  CHECK_EQ(index.check_invariants(), std::string());
  const float q[] = {0.3f};
  CHECK(index.search(q, 5, n).ids == brute_force_knn(base.data(), n, 1, q, 5).ids);
}

static void test_accessor_bounds() {
  const auto base = test::uniform_data(10, 2, 8);
  const HnswIndex index = test::build(base, 2, test::params(4, 10, 1));
  CHECK_THROWS(index.node_level(10), std::out_of_range, "out of range");
  CHECK_THROWS(index.neighbors(10, 0), std::out_of_range, "out of range");
  CHECK_THROWS(index.neighbors(0, -1), std::out_of_range, "has no level");
}

static void test_add_rejects_non_finite() {
  const size_t dim = 4;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const auto base = test::uniform_data(40, dim, 11);

  // Reference: the same adds without any failed attempt in between.
  test::TempFile ref_file("nonfinite_ref"), got_file("nonfinite_got");
  HnswIndex ref(dim, test::params(4, 20, 5));
  for (size_t i = 0; i < 40; ++i) ref.add(base.data() + i * dim);
  ref.save(ref_file.path());

  HnswIndex index(dim, test::params(4, 20, 5));
  const float bad[][4] = {{nan, 0, 0, 0}, {0, inf, 0, 0}, {0, 0, -inf, 0}, {0, 0, 0, nan}};
  const char* where[] = {"element 0 ", "element 1 ", "element 2 ", "element 3 "};
  for (size_t i = 0; i < 40; ++i) {
    if (i % 10 == 0) {
      // Fails on an empty index (i == 0) and on a populated one.
      for (int b = 0; b < 4; ++b) {
        const size_t before = index.size();
        CHECK_THROWS(index.add(bad[b]), std::invalid_argument, where[b]);
        CHECK_THROWS(index.add(bad[b]), std::invalid_argument, "not finite");
        CHECK_EQ(index.size(), before);
        CHECK_EQ(index.check_invariants(), std::string());
      }
    }
    CHECK_EQ(index.add(base.data() + i * dim), static_cast<uint32_t>(i));
  }
  // Failed adds changed nothing, not even the level-sampling RNG: the saved
  // file is byte-identical to the reference and loads under deep validation.
  index.save(got_file.path());
  CHECK(test::read_bytes(got_file.path()) == test::read_bytes(ref_file.path()));
  const LoadedIndex loaded = LoadedIndex::open(got_file.path());
  CHECK_EQ(loaded.size(), size_t{40});
}

int main() {
  test_constructor_validation();
  test_empty_index();
  test_one_and_two_elements();
  test_k_variants();
  test_duplicates();
  test_dim_one();
  test_accessor_bounds();
  test_add_rejects_non_finite();
  return test_finish();
}
