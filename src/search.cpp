#include <algorithm>
#include <functional>
#include <queue>

#include "fixed_heap.h"
#include "hnsw/graph.h"
#include "search_layer.h"

namespace hnsw {
namespace detail {
namespace {

#if HNSW_OPT_EPOCH_VISITED
// Epoch-tagged marks owned by the scratch object.
class Visited {
 public:
  Visited(SearchScratch& scratch, size_t n) : scratch_(scratch) { scratch_.begin_pass(n); }
  bool test_and_set(uint32_t id) { return scratch_.test_and_set(id); }

 private:
  SearchScratch& scratch_;
};
#else
// Baseline: a zeroed byte array allocated for every pass.
class Visited {
 public:
  Visited(SearchScratch&, size_t n) : marks_(n, 0) {}
  bool test_and_set(uint32_t id) {
    const bool seen = marks_[id] != 0;
    marks_[id] = 1;
    return seen;
  }

 private:
  std::vector<uint8_t> marks_;
};
#endif

#if HNSW_OPT_ALLOC_FREE_HEAPS
class CandidateHeap : public FixedHeap<true> {
 public:
  explicit CandidateHeap(SearchScratch& s) : FixedHeap<true>(s.candidate_buffer()) {}
};

class ResultHeap : public FixedHeap<false> {
 public:
  explicit ResultHeap(SearchScratch& s) : FixedHeap<false>(s.result_buffer()) {}
  // The buffer is scratch.result_buffer(); sorting it in place leaves the
  // results ascending there.
  void finish() { sort_in_place(); }
};
#else
// Baseline: std::priority_queue, allocating per pass.
class CandidateHeap
    : public std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> {
 public:
  explicit CandidateHeap(SearchScratch&) {}
};

class ResultHeap : public std::priority_queue<Cand> {
 public:
  explicit ResultHeap(SearchScratch& s) : out_(s.result_buffer()) {}
  // Moves the results, ascending, into scratch.result_buffer().
  void finish() {
    out_.resize(size());
    for (size_t i = out_.size(); i-- > 0;) {
      out_[i] = top();
      pop();
    }
  }

 private:
  std::vector<Cand>& out_;
};
#endif

}  // namespace

Cand greedy_closest(const GraphView& g, DistFn dist, const float* q, Cand cur, int level) {
  const size_t dim = g.layout.dim;
  bool improved = true;
  while (improved) {
    improved = false;
    for (uint32_t e : g.neighbors(cur.second, level)) {
      const Cand c{dist(q, g.vector(e), dim), e};
      if (c < cur) {
        cur = c;
        improved = true;
      }
    }
  }
  return cur;
}

CandidateSpan search_layer(const GraphView& g, DistFn dist, const float* q,
                           const Cand* entry, size_t num_entry, size_t ef, int level,
                           SearchScratch& scratch) {
  const size_t dim = g.layout.dim;
  scratch.reserve_heaps(g.count, ef);
  Visited visited(scratch, g.count);
  CandidateHeap candidates(scratch);
  ResultHeap found(scratch);
  for (size_t i = 0; i < num_entry; ++i) {
    const Cand& e = entry[i];
    if (visited.test_and_set(e.second)) continue;
    candidates.push(e);
    found.push(e);
    if (found.size() > ef) found.pop();
  }
  while (!candidates.empty()) {
    const Cand c = candidates.top();
    if (c.first > found.top().first) break;
    candidates.pop();
    for (uint32_t e : g.neighbors(c.second, level)) {
      if (visited.test_and_set(e)) continue;
      const Cand cand{dist(q, g.vector(e), dim), e};
      if (found.size() < ef || cand < found.top()) {
        candidates.push(cand);
        found.push(cand);
        if (found.size() > ef) found.pop();
      }
    }
  }
  found.finish();
  const std::vector<Cand>& out = scratch.result_buffer();
  return {out.data(), out.size()};
}

}  // namespace detail

size_t search_graph_into(const GraphView& g, DistFn dist, const float* query, size_t k,
                         size_t ef, SearchScratch& scratch, uint32_t* out_ids,
                         float* out_dists) {
  using detail::Cand;
  if (g.count == 0 || k == 0) return 0;

  Cand cur{dist(query, g.vector(g.entry_point), g.layout.dim), g.entry_point};
  for (int lc = g.max_level; lc > 0; --lc) cur = detail::greedy_closest(g, dist, query, cur, lc);
  const detail::CandidateSpan w =
      detail::search_layer(g, dist, query, &cur, 1, std::max(ef, k), 0, scratch);

  const size_t take = std::min(k, w.size);
  for (size_t i = 0; i < take; ++i) {
    out_dists[i] = w.data[i].first;
    out_ids[i] = w.data[i].second;
  }
  return take;
}

KnnResult search_graph(const GraphView& g, DistFn dist, const float* query, size_t k,
                       size_t ef, SearchScratch& scratch) {
  KnnResult result;
  const size_t cap = std::min(k, g.count);
  result.ids.resize(cap);
  result.dists.resize(cap);
  const size_t got = search_graph_into(g, dist, query, k, ef, scratch, result.ids.data(),
                                       result.dists.data());
  result.ids.resize(got);
  result.dists.resize(got);
  return result;
}

}  // namespace hnsw
