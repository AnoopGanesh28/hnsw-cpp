#pragma once

#include <cstddef>

namespace hnsw {

// Distance function signature. Callers hold a DistFn so that a SIMD
// implementation can replace the scalar one without changing call sites.
// Implementations return a value that orders vectors the same way the true
// distance does; for L2 this is the SQUARED Euclidean distance (no sqrt).
using DistFn = float (*)(const float* a, const float* b, size_t dim);

// Scalar reference implementation of squared L2 distance:
// sum_i (a[i] - b[i])^2. Returns the SQUARED distance, not its square root.
float l2_squared_scalar(const float* a, const float* b, size_t dim);

}  // namespace hnsw
