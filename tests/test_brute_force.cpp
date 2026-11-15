#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

#include "hnsw/brute_force.h"
#include "test_helper.h"

using namespace hnsw;

// Independent reference: double-precision distances and repeated min-selection
// (no sorting library), ties broken by lower id.
static KnnResult naive_knn(const std::vector<float>& base, size_t n, size_t dim,
                           const float* q, size_t k) {
  std::vector<double> d(n);
  for (size_t i = 0; i < n; ++i) {
    double s = 0.0;
    for (size_t j = 0; j < dim; ++j) {
      const double diff = static_cast<double>(base[i * dim + j]) - q[j];
      s += diff * diff;
    }
    d[i] = s;
  }
  std::vector<bool> used(n, false);
  KnnResult r;
  for (size_t t = 0; t < std::min(k, n); ++t) {
    size_t best = n;
    for (size_t i = 0; i < n; ++i) {
      if (used[i]) continue;
      if (best == n || d[i] < d[best]) best = i;  // strict <: lowest id wins ties
    }
    used[best] = true;
    r.ids.push_back(static_cast<uint32_t>(best));
    r.dists.push_back(static_cast<float>(d[best]));
  }
  return r;
}

static std::vector<float> random_data(size_t n, size_t dim, unsigned seed, bool integral) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::uniform_int_distribution<int> iu(0, 3);
  std::vector<float> v(n * dim);
  for (float& x : v) x = integral ? static_cast<float>(iu(rng)) : u(rng);
  return v;
}

static void compare(const KnnResult& got, const KnnResult& want, double eps) {
  CHECK_EQ(got.ids.size(), want.ids.size());
  CHECK_EQ(got.dists.size(), want.dists.size());
  if (got.ids.size() != want.ids.size()) return;
  CHECK(got.ids == want.ids);
  for (size_t i = 0; i < got.dists.size(); ++i) {
    CHECK_NEAR(got.dists[i], want.dists[i], eps);
  }
  for (size_t i = 1; i < got.dists.size(); ++i) {
    CHECK(got.dists[i - 1] <= got.dists[i]);
  }
}

static void test_random_float() {
  const size_t n = 300, dim = 16;
  const auto base = random_data(n, dim, 7, false);
  const auto queries = random_data(20, dim, 8, false);
  for (size_t k : {1u, 5u, 10u, 50u}) {
    for (size_t q = 0; q < 20; ++q) {
      const float* qp = queries.data() + q * dim;
      compare(brute_force_knn(base.data(), n, dim, qp, k),
              naive_knn(base, n, dim, qp, k), 1e-4);
    }
  }
}

static void test_k_edges() {
  const size_t n = 40, dim = 5;
  const auto base = random_data(n, dim, 3, false);
  const auto q = random_data(1, dim, 4, false);

  // k == n
  compare(brute_force_knn(base.data(), n, dim, q.data(), n),
          naive_knn(base, n, dim, q.data(), n), 1e-4);

  // k > n returns exactly n results.
  const KnnResult over = brute_force_knn(base.data(), n, dim, q.data(), n + 25);
  CHECK_EQ(over.ids.size(), n);
  compare(over, naive_knn(base, n, dim, q.data(), n + 25), 1e-4);

  // k == 0 and empty base give empty results.
  CHECK_EQ(brute_force_knn(base.data(), n, dim, q.data(), 0).ids.size(), size_t{0});
  const KnnResult empty = brute_force_knn(base.data(), 0, dim, q.data(), 5);
  CHECK_EQ(empty.ids.size(), size_t{0});
  CHECK_EQ(empty.dists.size(), size_t{0});
}

static void test_ties() {
  // Integer coordinates in {0..3} with dim 2 force many exact distance ties;
  // float arithmetic is exact here so ids and distances must match exactly.
  const size_t n = 200, dim = 2;
  const auto base = random_data(n, dim, 11, true);
  const auto queries = random_data(15, dim, 12, true);
  for (size_t q = 0; q < 15; ++q) {
    const float* qp = queries.data() + q * dim;
    for (size_t k : {3u, 17u, 200u}) {
      compare(brute_force_knn(base.data(), n, dim, qp, k),
              naive_knn(base, n, dim, qp, k), 0.0);
    }
  }

  // Identical vectors: the result must be ascending ids.
  const std::vector<float> same(6 * 3, 1.0f);
  const float q[] = {1.0f, 1.0f, 1.0f};
  const KnnResult r = brute_force_knn(same.data(), 6, 3, q, 4);
  CHECK((r.ids == std::vector<uint32_t>{0, 1, 2, 3}));
  for (float d : r.dists) CHECK_EQ(d, 0.0f);
}

static void test_batch_matches_single() {
  const size_t n = 150, dim = 8, nq = 12, k = 9;
  const auto base = random_data(n, dim, 21, false);
  const auto queries = random_data(nq, dim, 22, false);
  const auto batch = brute_force_knn_batch(base.data(), n, dim, queries.data(), nq, k);
  CHECK_EQ(batch.size(), nq);
  for (size_t q = 0; q < nq && q < batch.size(); ++q) {
    const KnnResult single = brute_force_knn(base.data(), n, dim, queries.data() + q * dim, k);
    CHECK(batch[q].ids == single.ids);
    CHECK(batch[q].dists == single.dists);
  }
  CHECK_EQ(brute_force_knn_batch(base.data(), n, dim, queries.data(), 0, k).size(), size_t{0});
}

static float negated_dot(const float* a, const float* b, size_t dim) {
  float s = 0.0f;
  for (size_t i = 0; i < dim; ++i) s -= a[i] * b[i];
  return s;
}

static void test_custom_dist_fn() {
  const float base[] = {1, 0, 2, 0, 3, 0};  // dim 2, vectors (1,0) (2,0) (3,0)
  const float q[] = {1, 0};
  const KnnResult r = brute_force_knn(base, 3, 2, q, 2, negated_dot);
  CHECK((r.ids == std::vector<uint32_t>{2, 1}));
}

int main() {
  test_random_float();
  test_k_edges();
  test_ties();
  test_batch_matches_single();
  test_custom_dist_fn();
  return test_finish();
}
