#pragma once

// Shared by golden_gen.cpp (built against the M2 baseline by gen_golden.sh)
// and test_golden.cpp. Uses only the public API present since M2, and always
// the scalar kernel, so both builds compute the same hashes when graphs and
// results are bit-identical.

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "hnsw/hnsw.h"

namespace golden {

struct Config {
  const char* name;
  size_t n, dim, M, ef_construction;
  uint64_t seed;
  bool keep_pruned;
  bool integer_grid;  // coordinates in {0..3}: forces many exact distance ties
};

inline constexpr Config kConfigs[] = {
    {"uniform_d16_M8", 1500, 16, 8, 100, 1, false, false},
    {"uniform_d32_M16", 1500, 32, 16, 200, 2, false, false},
    {"uniform_d5_M4_keep_pruned", 800, 5, 4, 50, 3, true, false},
    {"grid_d3_M4_ties", 600, 3, 4, 50, 4, false, true},
    {"uniform_d1_M2", 400, 1, 2, 40, 5, false, false},
};
inline constexpr size_t kQueries = 100;
inline constexpr size_t kK = 10;
inline constexpr size_t kEfs[] = {10, 64};

inline std::vector<float> make_data(size_t n, size_t dim, unsigned seed, bool grid) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::uniform_int_distribution<int> iu(0, 3);
  std::vector<float> v(n * dim);
  for (float& x : v) x = grid ? static_cast<float>(iu(rng)) : u(rng);
  return v;
}

// FNV-1a, 64-bit.
struct Hasher {
  uint64_t h = 1469598103934665603ull;
  void bytes(const void* p, size_t n) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) {
      h ^= b[i];
      h *= 1099511628211ull;
    }
  }
  void u32(uint32_t x) { bytes(&x, sizeof(x)); }
  void f32(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    u32(bits);
  }
};

struct Hashes {
  uint64_t graph;
  uint64_t search;
};

inline Hashes compute(const Config& c) {
  const std::vector<float> base = make_data(c.n, c.dim, static_cast<unsigned>(c.seed), c.integer_grid);
  const std::vector<float> queries =
      make_data(kQueries, c.dim, static_cast<unsigned>(c.seed) + 1000, c.integer_grid);
  hnsw::HnswParams p;
  p.M = c.M;
  p.ef_construction = c.ef_construction;
  p.seed = c.seed;
  p.keep_pruned = c.keep_pruned;
  hnsw::HnswIndex index(c.dim, p, hnsw::l2_squared_scalar);
  for (size_t i = 0; i < c.n; ++i) index.add(base.data() + i * c.dim);

  Hasher g;
  g.u32(static_cast<uint32_t>(index.max_level()));
  g.u32(index.entry_point());
  for (uint32_t id = 0; id < index.size(); ++id) {
    const int level = index.node_level(id);
    g.u32(static_cast<uint32_t>(level));
    for (int l = 0; l <= level; ++l) {
      const hnsw::NeighborView nb = index.neighbors(id, l);
      g.u32(static_cast<uint32_t>(nb.count));
      for (uint32_t x : nb) g.u32(x);
    }
  }

  Hasher s;
  for (size_t q = 0; q < kQueries; ++q) {
    for (size_t ef : kEfs) {
      const hnsw::KnnResult r = index.search(queries.data() + q * c.dim, kK, ef);
      s.u32(static_cast<uint32_t>(r.ids.size()));
      for (size_t i = 0; i < r.ids.size(); ++i) {
        s.u32(r.ids[i]);
        s.f32(r.dists[i]);
      }
    }
  }
  return {g.h, s.h};
}

}  // namespace golden
