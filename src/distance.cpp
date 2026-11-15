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

}  // namespace hnsw
