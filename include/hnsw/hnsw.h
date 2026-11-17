#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "hnsw/aligned_buffer.h"
#include "hnsw/brute_force.h"
#include "hnsw/distance.h"
#include "hnsw/graph.h"
#include "hnsw/layout.h"
#include "hnsw/scratch.h"

namespace hnsw {

struct HnswParams {
  // Max neighbors per node on layers >= 1; layer 0 allows 2 * M.
  size_t M = 16;
  // Candidate list size used while inserting.
  size_t ef_construction = 200;
  // Seed for level sampling. Same seed + same insertion order => same graph.
  uint64_t seed = 100;
  // Algorithm 4 keepPrunedConnections: fill up to M with discarded candidates.
  bool keep_pruned = false;
};

// Hierarchical Navigable Small World index (Malkov & Yashunin). Ids are
// assigned sequentially from 0 in insertion order. Insertion is single
// threaded. While no add() or reserve() runs, the search methods are safe to
// call from any number of threads at once, provided each thread uses its own
// SearchScratch (the overloads without one use a thread_local scratch).
// Storage follows Layout: one contiguous arena of fixed-size layer-0 records
// (links + vector) plus a separate table for upper-layer links.
class HnswIndex {
 public:
  // Throws std::invalid_argument if dim < 1, M < 2, ef_construction < 1, or
  // dist is null. The default kernel is the fastest one for this CPU.
  HnswIndex(size_t dim, HnswParams params = {}, DistFn dist = best_l2_squared());

  // Preallocates storage for n elements in total.
  void reserve(size_t n);

  // Copies dim floats from vec into the index and links it. Returns its id.
  // Throws std::invalid_argument, before changing any state, if any element is
  // NaN or infinite (the file loader rejects such vectors, so an index holding
  // one could be saved but never reloaded). Throws std::length_error if the
  // index already holds 2^32 - 1 elements.
  uint32_t add(const float* vec);

  // Approximate k nearest neighbors, ascending by distance, ties by lower id.
  // Uses a candidate list of size max(ef, k). Returns at most min(k, size())
  // results; empty if the index is empty or k == 0.
  // This overload uses a thread_local SearchScratch.
  KnnResult search(const float* query, size_t k, size_t ef) const;
  // Same, with caller-owned scratch (one per thread).
  KnnResult search(const float* query, size_t k, size_t ef, SearchScratch& scratch) const;
  // Allocation-free form (see search_graph_into): writes up to min(k, size())
  // results into out_ids / out_dists and returns how many were written.
  size_t search_into(const float* query, size_t k, size_t ef, SearchScratch& scratch,
                     uint32_t* out_ids, float* out_dists) const;
  // Batch search on `pool` (see search_graph_batch): writes n_queries rows of
  // k results into out_ids / out_dists, padded with kNoNeighbor / +infinity.
  void search_batch(const float* queries, size_t n_queries, size_t k, size_t ef,
                    uint32_t* out_ids, float* out_dists, ThreadPool& pool,
                    size_t chunk = kDefaultBatchChunk) const;

  size_t size() const { return levels_.size(); }
  size_t dim() const { return dim_; }
  const HnswParams& params() const { return params_; }
  const Layout& layout() const { return layout_; }

  // Read-only view of the current graph; invalidated by add() and reserve().
  GraphView view() const;

  // Highest layer in the graph; -1 when the index is empty.
  int max_level() const { return max_level_; }
  // Node searches start from. Throws std::logic_error when empty.
  uint32_t entry_point() const;
  // Top layer of node id. Throws std::out_of_range for an invalid id.
  int node_level(uint32_t id) const;
  // Neighbors of id on `level`. Throws std::out_of_range unless
  // id < size() and 0 <= level <= node_level(id).
  NeighborView neighbors(uint32_t id, int level) const;

  // Empty string if every structural invariant holds, otherwise a
  // description of the first violation found.
  std::string check_invariants() const;

  // Writes the index to `path` in the format of docs/FORMAT.md (load it with
  // LoadedIndex::open). The distance kernel is not stored: the file records
  // the metric as squared L2, so an index built with another DistFn must not
  // be saved. Writes a temporary file in the same directory, flushes it to
  // disk, and renames it over `path`, so readers see either the old file or
  // the complete new one. Throws IndexFileError on any I/O error (leaving no
  // temporary file behind) or if dim or M exceed the format limits.
  void save(const std::string& path) const;

 private:
  using Cand = Candidate;

  const float* vec(uint32_t id) const;
  uint32_t* link_block(uint32_t id, int level);
  float distance(const float* q, uint32_t id) const { return dist_(q, vec(id), dim_); }

  int sample_level();
  std::vector<uint32_t> select_neighbors(const std::vector<Cand>& sorted, size_t m) const;
  void set_links(uint32_t id, int level, const std::vector<uint32_t>& ids);
  void connect(uint32_t from, uint32_t to, int level);

  size_t dim_;
  HnswParams params_;
  DistFn dist_;
  Layout layout_;
  double ml_ = 0.0;
  std::mt19937_64 rng_;

  AlignedBuffer arena_;                 // size() * record_stride bytes
  std::vector<uint8_t> levels_;         // top layer per node
  std::vector<uint64_t> upper_offsets_;  // word offset into upper_links_ (level >= 1 only)
  std::vector<uint32_t> upper_links_;
  int max_level_ = -1;
  uint32_t entry_ = 0;
  SearchScratch insert_scratch_;
};

}  // namespace hnsw
