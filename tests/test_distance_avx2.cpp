// Randomized comparison of l2_squared_avx2 against the scalar reference and
// a double-precision reference. Skips (exit 77) without AVX2 + FMA.

#include <cmath>
#include <random>
#include <vector>

#include "hnsw/distance.h"
#include "test_helper.h"

using namespace hnsw;

// Error bound. Each term t_i = (a_i - b_i)^2 is nonnegative. In float, the
// subtraction and the multiply (or FMA) each contribute one rounding factor
// (1 + delta), |delta| <= u = 2^-24, and every addition on the path from a term
// to the result contributes one more. In l2_squared_avx2 a term passes through
// at most dim/32 accumulator FMAs, 2 accumulator combines, 3 horizontal adds
// and dim % 8 tail adds; in the scalar loop through at most dim - 1 adds. In
// both cases fewer than dim + 3 factors, so by Higham's lemma
// (prod of k factors (1 + delta)^{+-1} = 1 + theta_k, |theta_k| <= gamma_k)
//   |computed - S| <= gamma_{dim+3} * sum_i t_i = gamma_{dim+3} * S,
// with gamma_k = k u / (1 - k u) and S the exact sum (all terms >= 0, so the
// sum of |t_i| is S itself; no cancellation). The double reference has the
// same bound with u = 2^-53, which is added. Magnitudes here (>= 1e-3) keep
// all terms far from the subnormal range, so no absolute term is needed.
static double gamma(size_t k, double u) { return k * u / (1.0 - k * u); }

static double bound(size_t dim, double s) {
  return (gamma(dim + 3, std::ldexp(1.0, -24)) + gamma(dim + 3, std::ldexp(1.0, -53))) * s;
}

static double reference(const float* a, const float* b, size_t dim) {
  double s = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    s += d * d;
  }
  return s;
}

static void test_random_all_dims() {
  std::mt19937 rng(12345);
  bool ok = true;
  for (size_t dim = 1; dim <= 300; ++dim) {
    for (int exponent = -3; exponent <= 3; ++exponent) {
      const float scale = static_cast<float>(std::pow(10.0, exponent));
      std::uniform_real_distribution<float> u(-scale, scale);
      for (int rep = 0; rep < 4; ++rep) {
        // One extra float so both operands can be offset by one (misaligned).
        std::vector<float> a(dim + 1), b(dim + 1);
        for (float& x : a) x = u(rng);
        for (float& x : b) x = u(rng);
        const size_t off = static_cast<size_t>(rep % 2);
        const float* pa = a.data() + off;
        const float* pb = b.data() + off;
        const double ref = reference(pa, pb, dim);
        const double eps = bound(dim, ref);
        const double simd = l2_squared_avx2(pa, pb, dim);
        const double scalar = l2_squared_scalar(pa, pb, dim);
        ok = ok && std::fabs(simd - ref) <= eps;
        ok = ok && std::fabs(scalar - ref) <= eps;
        ok = ok && std::fabs(simd - scalar) <= 2 * eps;
        if (!ok) {
          CHECK_NEAR(simd, ref, eps);
          CHECK_NEAR(scalar, ref, eps);
          return;
        }
      }
    }
  }
  CHECK(ok);
}

static void test_exact_cases() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-1000.0f, 1000.0f);
  bool ok = true;
  for (size_t dim = 1; dim <= 300; ++dim) {
    std::vector<float> a(dim + 1), zeros(dim + 1, 0.0f);
    for (float& x : a) x = u(rng);
    const std::vector<float> copy = a;
    for (size_t off : {0u, 1u}) {
      // Identical vectors: every difference is exactly 0.
      ok = ok && l2_squared_avx2(a.data() + off, copy.data() + off, dim) == 0.0f;
      ok = ok && l2_squared_avx2(a.data() + off, a.data() + off, dim) == 0.0f;
      ok = ok && l2_squared_avx2(zeros.data() + off, zeros.data() + off, dim) == 0.0f;
    }
    // Small integers: all partial sums are exact in float, so both kernels
    // must agree exactly with the integer result.
    std::vector<float> p(dim), q(dim);
    long long expect = 0;
    for (size_t i = 0; i < dim; ++i) {
      p[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
      q[i] = static_cast<float>(static_cast<int>(i % 5) - 2);
      const long long d = static_cast<long long>(p[i] - q[i]);
      expect += d * d;
    }
    ok = ok && l2_squared_avx2(p.data(), q.data(), dim) == static_cast<float>(expect);
    ok = ok && l2_squared_scalar(p.data(), q.data(), dim) == static_cast<float>(expect);
  }
  CHECK(ok);
  const float a[] = {1.0f}, b[] = {2.0f};
  CHECK_EQ(l2_squared_avx2(a, b, 0), 0.0f);
}

// Each element contributes: setting a single coordinate catches any element
// skipped by the main loops or the tail, for every dim and position.
static void test_every_position_counts() {
  bool ok = true;
  for (size_t dim = 1; dim <= 300; ++dim) {
    const std::vector<float> zeros(dim, 0.0f);
    for (size_t pos = 0; pos < dim; ++pos) {
      std::vector<float> a(dim, 0.0f);
      a[pos] = 3.0f;
      ok = ok && l2_squared_avx2(a.data(), zeros.data(), dim) == 9.0f;
    }
  }
  CHECK(ok);
}

static void test_dispatch() {
  CHECK(best_l2_squared() == l2_squared_avx2);
}

int main() {
  if (!cpu_supports_avx2_fma()) return 77;
  test_random_all_dims();
  test_exact_cases();
  test_every_position_counts();
  test_dispatch();
  return test_finish();
}
