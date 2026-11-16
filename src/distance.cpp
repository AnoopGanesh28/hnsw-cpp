#include "hnsw/distance.h"

namespace hnsw {

float l2_squared_scalar(const float* a, const float* b, size_t dim) {
  float sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

bool cpu_supports_avx2_fma() {
  static const bool supported = [] {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  }();
  return supported;
}

DistFn best_l2_squared() {
  return cpu_supports_avx2_fma() ? l2_squared_avx2 : l2_squared_scalar;
}

}  // namespace hnsw
