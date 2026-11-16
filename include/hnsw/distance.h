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

// AVX2 + FMA squared L2 distance. Accepts unaligned pointers and any dim.
// Only call it when cpu_supports_avx2_fma() is true. Results can differ from
// the scalar reference by rounding (different summation order and FMA).
float l2_squared_avx2(const float* a, const float* b, size_t dim);

// True if the running CPU supports both AVX2 and FMA.
bool cpu_supports_avx2_fma();

// Fastest squared-L2 kernel for the running CPU: l2_squared_avx2 when
// supported, otherwise l2_squared_scalar. The default kernel of HnswIndex
// and brute-force search.
DistFn best_l2_squared();

}  // namespace hnsw
