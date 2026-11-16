#pragma once

// Internal: binary heap over a caller-owned std::vector whose capacity has
// been reserved for the maximum size, so push/pop never allocate.

#include <cstddef>
#include <utility>
#include <vector>

#include "hnsw/scratch.h"

namespace hnsw {
namespace detail {

// kMinHeap: top() is the smallest Candidate; otherwise the largest.
// Candidates order by (distance, id), as everywhere else.
template <bool kMinHeap>
class FixedHeap {
 public:
  explicit FixedHeap(std::vector<Candidate>& buffer) : h_(buffer) { h_.clear(); }

  bool empty() const { return h_.empty(); }
  size_t size() const { return h_.size(); }
  const Candidate& top() const { return h_[0]; }

  void push(const Candidate& c) {
    h_.push_back(c);
    sift_up(h_.size() - 1);
  }

  void pop() {
    h_[0] = h_.back();
    h_.pop_back();
    if (!h_.empty()) sift_down(0, h_.size());
  }

  // In-place heapsort. For a max-heap this leaves the buffer ascending; the
  // heap must not be used afterwards.
  void sort_in_place() {
    for (size_t end = h_.size(); end > 1; --end) {
      std::swap(h_[0], h_[end - 1]);
      sift_down(0, end - 1);
    }
  }

 private:
  // True if a belongs above b.
  static bool above(const Candidate& a, const Candidate& b) { return kMinHeap ? a < b : b < a; }

  void sift_up(size_t i) {
    while (i > 0) {
      const size_t parent = (i - 1) / 2;
      if (!above(h_[i], h_[parent])) break;
      std::swap(h_[i], h_[parent]);
      i = parent;
    }
  }

  void sift_down(size_t i, size_t n) {
    for (;;) {
      const size_t left = 2 * i + 1;
      if (left >= n) break;
      size_t best = left;
      if (left + 1 < n && above(h_[left + 1], h_[left])) best = left + 1;
      if (!above(h_[best], h_[i])) break;
      std::swap(h_[i], h_[best]);
      i = best;
    }
  }

  std::vector<Candidate>& h_;
};

}  // namespace detail
}  // namespace hnsw
