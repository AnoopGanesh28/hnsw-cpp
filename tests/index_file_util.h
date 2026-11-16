#pragma once

// Helpers for index file tests. Field offsets and the CRC are written out
// from docs/FORMAT.md independently of src/index_file.cpp, so the tests check
// the library against the spec rather than against itself.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "hnsw/hnsw.h"
#include "hnsw/index_file.h"
#include "test_helper.h"

namespace test {

namespace hdr {
constexpr size_t kVersion = 8, kEndian = 12, kMetric = 16, kDim = 20, kM = 24, kCount = 32,
                 kEntry = 40, kMaxLevel = 44, kStride = 48, kFileSize = 56, kSections = 64,
                 kReserved = 128, kCrc = 188, kSize = 192;
// Section i: offset at kSections + 16 * i, length at kSections + 16 * i + 8.
enum Section { kLevels, kArena, kUpperOffsets, kUpperLinks };
}  // namespace hdr

// Bitwise CRC-32 (IEEE, reflected 0xEDB88320), independent of the library's
// table-driven one.
inline uint32_t crc32_bitwise(const uint8_t* p, size_t n) {
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) {
    c ^= p[i];
    for (int b = 0; b < 8; ++b) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
  }
  return ~c;
}

template <typename T>
T get_field(const std::vector<uint8_t>& f, size_t off) {
  T v;
  std::memcpy(&v, f.data() + off, sizeof(v));
  return v;
}
template <typename T>
void set_field(std::vector<uint8_t>& f, size_t off, T v) {
  std::memcpy(f.data() + off, &v, sizeof(v));
}
inline uint64_t section_offset(const std::vector<uint8_t>& f, int s) {
  return get_field<uint64_t>(f, hdr::kSections + 16 * s);
}
inline uint64_t section_bytes(const std::vector<uint8_t>& f, int s) {
  return get_field<uint64_t>(f, hdr::kSections + 16 * s + 8);
}
inline void fix_crc(std::vector<uint8_t>& f) {
  set_field<uint32_t>(f, hdr::kCrc, crc32_bitwise(f.data(), hdr::kCrc));
}

inline std::vector<uint8_t> read_bytes(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
inline void write_bytes(const std::string& path, const std::vector<uint8_t>& b, size_t n) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(n));
}
inline void write_bytes(const std::string& path, const std::vector<uint8_t>& b) {
  write_bytes(path, b, b.size());
}

// Floats k * 2^-24 from integer draws: exact, no float arithmetic beyond one
// exact scaling, so the data is identical under any compiler and flags.
inline std::vector<float> exact_data(size_t n, size_t dim, unsigned seed) {
  std::mt19937 rng(seed);
  std::vector<float> v(n * dim);
  for (float& x : v) x = static_cast<float>(rng() >> 8) * 0x1.0p-24f;
  return v;
}

inline hnsw::HnswIndex build_exact(size_t n, size_t dim, size_t M, size_t efc, uint64_t seed,
                                   hnsw::DistFn dist) {
  hnsw::HnswParams p;
  p.M = M;
  p.ef_construction = efc;
  p.seed = seed;
  hnsw::HnswIndex index(dim, p, dist);
  const std::vector<float> data = exact_data(n, dim, static_cast<unsigned>(seed));
  for (size_t i = 0; i < n; ++i) index.add(data.data() + i * dim);
  return index;
}

// Runs a few searches; used to show a loaded (possibly corrupted) index is
// safe to query. Returns the number of results, so the work is not elided.
inline size_t probe_searches(const hnsw::LoadedIndex& index) {
  const std::vector<float> queries = exact_data(4, index.dim(), 999);
  size_t total = 0;
  for (size_t q = 0; q < 4; ++q) {
    total += index.search(queries.data() + q * index.dim(), 5, 16).ids.size();
  }
  return total;
}

}  // namespace test
