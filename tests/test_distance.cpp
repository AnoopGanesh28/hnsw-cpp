#include <random>
#include <vector>

#include "hnsw/distance.h"
#include "test_helper.h"

using hnsw::DistFn;
using hnsw::l2_squared_scalar;

static void test_hand_computed() {
  const float a1[] = {3.0f}, b1[] = {1.0f};
  CHECK_EQ(l2_squared_scalar(a1, b1, 1), 4.0f);

  const float a3[] = {1, 2, 3}, b3[] = {4, 6, 8};  // 9 + 16 + 25
  CHECK_EQ(l2_squared_scalar(a3, b3, 3), 50.0f);

  const float a7[] = {0, 1, 2, 3, 4, 5, 6}, b7[] = {1, 2, 3, 4, 5, 6, 7};
  CHECK_EQ(l2_squared_scalar(a7, b7, 7), 7.0f);

  std::vector<float> a33(33), b33(33, 0.0f);  // sum_{i=0}^{32} i^2 = 11440
  for (int i = 0; i < 33; ++i) a33[i] = static_cast<float>(i);
  CHECK_EQ(l2_squared_scalar(a33.data(), b33.data(), 33), 11440.0f);

  const float neg_a[] = {-1.5f, 2.0f}, neg_b[] = {1.5f, -2.0f};  // 9 + 16
  CHECK_EQ(l2_squared_scalar(neg_a, neg_b, 2), 25.0f);
}

static void test_zero_and_identical() {
  for (size_t dim : {1u, 3u, 7u, 33u, 128u}) {
    std::vector<float> zeros(dim, 0.0f);
    CHECK_EQ(l2_squared_scalar(zeros.data(), zeros.data(), dim), 0.0f);

    std::mt19937 rng(static_cast<unsigned>(dim));
    std::uniform_real_distribution<float> u(-10.0f, 10.0f);
    std::vector<float> v(dim), copy;
    for (float& x : v) x = u(rng);
    copy = v;
    CHECK_EQ(l2_squared_scalar(v.data(), copy.data(), dim), 0.0f);
    CHECK_EQ(l2_squared_scalar(v.data(), v.data(), dim), 0.0f);
  }
  const float a[] = {1.0f}, b[] = {2.0f};
  CHECK_EQ(l2_squared_scalar(a, b, 0), 0.0f);
}

static void test_symmetry_and_double_reference() {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  for (size_t dim : {1u, 3u, 7u, 33u, 128u}) {
    std::vector<float> a(dim), b(dim);
    for (float& x : a) x = u(rng);
    for (float& x : b) x = u(rng);
    double ref = 0.0;
    for (size_t i = 0; i < dim; ++i) {
      const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
      ref += d * d;
    }
    const float ab = l2_squared_scalar(a.data(), b.data(), dim);
    CHECK_EQ(ab, l2_squared_scalar(b.data(), a.data(), dim));
    CHECK_NEAR(ab, ref, 1e-4 * (1.0 + ref));
  }
}

static void test_function_pointer() {
  DistFn fn = l2_squared_scalar;
  const float a[] = {0, 0}, b[] = {3, 4};
  CHECK_EQ(fn(a, b, 2), 25.0f);  // squared: 5^2, not 5
}

int main() {
  test_hand_computed();
  test_zero_and_identical();
  test_symmetry_and_double_reference();
  test_function_pointer();
  return test_finish();
}
