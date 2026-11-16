#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "json.h"

namespace bench {

// The hnswlib baseline (C++ header-only library, L2Space, float), wrapped so
// that hnswlib's headers are compiled in exactly one translation unit
// (hnswlib_index.cpp). That matters for hnswlib_info(): hnswlib selects its
// SIMD kernel by writing a file-static function pointer, so the selection can
// only be observed from the translation unit that uses it.
//
// Labels are the insertion index, so labels equal internal ids.
class HnswlibIndex {
 public:
  // Empty index for building, with hnswlib's own parameters
  // (max_elements, M, ef_construction, random_seed).
  HnswlibIndex(size_t dim, size_t max_elements, size_t M, size_t ef_construction,
               size_t seed);
  // Loads a file written by save() (hnswlib's loadIndex).
  static std::unique_ptr<HnswlibIndex> load(const std::string& path, size_t dim);
  ~HnswlibIndex();
  HnswlibIndex(const HnswlibIndex&) = delete;
  HnswlibIndex& operator=(const HnswlibIndex&) = delete;

  // hnswlib addPoint with label `label`.
  void add(const float* vec, size_t label);
  // hnswlib saveIndex.
  void save(const std::string& path);
  // hnswlib setEf. Not thread-safe: call it while no search runs.
  void set_ef(size_t ef);
  // hnswlib searchKnn (thread-safe), converted from hnswlib's max-heap to
  // ascending order: writes up to k results to ids / dists and returns how
  // many were written.
  size_t search_into(const float* query, size_t k, uint32_t* ids, float* dists) const;

  size_t size() const;

 private:
  struct Impl;
  explicit HnswlibIndex(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// What hnswlib's headers selected under this build's flags: the USE_SSE /
// USE_AVX / USE_AVX512 macros its headers define, its runtime CPU checks, and
// the L2 distance function an L2Space of dimension `dim` actually uses.
json::Value hnswlib_info(size_t dim);

}  // namespace bench
