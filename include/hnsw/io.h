#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace hnsw {

// A set of equal-dimension vectors stored flat, row-major:
// vector i occupies data[i * dim, (i + 1) * dim).
template <typename T>
struct VecSet {
  std::vector<T> data;
  size_t dim = 0;
  size_t count = 0;
};

using FvecsData = VecSet<float>;
using IvecsData = VecSet<int32_t>;

// fvecs/ivecs: each record is a little-endian int32 dimension followed by
// that many 4-byte little-endian values. All records must share one dimension.
//
// Readers throw std::runtime_error with a descriptive message if the file
// cannot be opened, is empty, has a non-positive dimension, has a record whose
// dimension differs from the first, or ends in the middle of a record.
FvecsData read_fvecs(const std::string& path);
IvecsData read_ivecs(const std::string& path);

// Writers take flat row-major data; data.size() must be a positive multiple of
// dim and dim must be positive, otherwise std::invalid_argument is thrown.
// Throws std::runtime_error if the file cannot be written.
void write_fvecs(const std::string& path, const std::vector<float>& data,
                 size_t dim);
void write_ivecs(const std::string& path, const std::vector<int32_t>& data,
                 size_t dim);

}  // namespace hnsw
