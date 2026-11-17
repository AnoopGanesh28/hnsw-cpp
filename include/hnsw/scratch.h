#pragma once

// SearchScratch: per-thread reusable search state (visited marks and heaps).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "hnsw/config.h"

namespace hnsw {

// (distance, id). Ordered by distance, then id: a total order, so traversal
// and results are deterministic under distance ties.
using Candidate = std::pair<float, uint32_t>;

// Per-thread mutable state for searching a graph. One object may be reused
// for any number of queries and graphs, but not by two threads at once.
//
// Visited marks are epoch tags: a node is visited in the current pass iff its
// tag equals the current epoch, so starting a pass is O(1) instead of
// clearing an n-byte array. Tags are 16-bit: half the memory (and cache
// footprint) of 32-bit tags, at the cost of one full clear every 65535
// passes, which is negligible next to 65535 searches.
//
// It also owns the buffers behind the search heaps. Their capacity only grows
// (to the graph size for candidates, min(ef + 1, size) for results), so once a
// scratch has served a query of a given graph size and ef, later such queries
// allocate nothing.
class SearchScratch {
 public:
  using Tag = uint16_t;

  // Starts a new search pass over a graph of n nodes: grows the tag array if
  // the graph has grown and advances the epoch. On wraparound all tags are
  // cleared and the epoch restarts at 1 (tag 0 is never a live epoch).
  void begin_pass(size_t n) {
    if (tags_.size() < n) tags_.resize(n, 0);
    if (++epoch_ == 0) {
      std::fill(tags_.begin(), tags_.end(), Tag{0});
      epoch_ = 1;
    }
  }

  // Returns whether id was already visited in this pass, and marks it.
  bool test_and_set(uint32_t id) {
    const bool seen = tags_[id] == epoch_;
    tags_[id] = epoch_;
    return seen;
  }

  const Tag* tag_address(uint32_t id) const { return tags_.data() + id; }

  // Ensures heap capacity for a pass over n nodes with result size ef.
  void reserve_heaps(size_t n, size_t ef) {
    candidates_.reserve(n);
    results_.reserve(std::min(ef + 1, n));
  }
  // Buffers used by the search routines.
  std::vector<Candidate>& candidate_buffer() { return candidates_; }
  std::vector<Candidate>& result_buffer() { return results_; }

  Tag epoch() const { return epoch_; }
  // Test hook: lets tests drive the epoch to its wraparound point.
  void set_epoch_for_testing(Tag epoch) { epoch_ = epoch; }

 private:
  std::vector<Tag> tags_;
  Tag epoch_ = 0;
  std::vector<Candidate> candidates_;
  std::vector<Candidate> results_;
};

}  // namespace hnsw
